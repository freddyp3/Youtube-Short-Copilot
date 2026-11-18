"""Sign-in for real users: the server is its own OAuth 2.1 authorization server and Google says who the user is.

Claude (or any MCP client) registers itself (dynamic client registration), sends the user to ``/authorize`` and
gets an access token from ``/token``; those endpoints come from the MCP SDK and call ``Provider`` below. In
between, the user signs in with Google and lands on ``/oauth/google/callback``:

    /authorize -> Google -> /oauth/google/callback -> (approved?) -> consent page -> /oauth/consent -> client

A first sign-in creates the user (that is the signup; ``users.created_at``). Nobody can use the tools before the
owner approves them (``python -m src.server.users approve EMAIL``), except the addresses in
``AUTO_APPROVE_EMAILS``. The consent page names the connecting client and needs a click, so a link built by
somebody else cannot quietly connect their client to a signed-in user's account.

Stored: the user's Google subject id, email and name; clients; and only SHA-256 hashes of authorization codes,
access tokens and refresh tokens. A user's id (``u_`` + 8 hex characters) is what drafts, quotas and log lines
carry, in the place of the token id of an invite token.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from src.server.store import connect, init_schema, transaction

SEEN_EVERY_S = 60.0
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
CALLBACK_PATH, CONSENT_PATH = "/oauth/google/callback", "/oauth/consent"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    google_sub TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    approved_at REAL,
    blocked_at REAL,
    last_seen_at REAL
);
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id TEXT PRIMARY KEY,
    info_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_pending (
    state_sha TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    expires_at REAL NOT NULL,
    user_id TEXT,
    ticket_sha TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS oauth_codes (
    code_sha TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_tokens (
    token_sha TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    grant_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scopes TEXT NOT NULL,
    resource TEXT,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS oauth_tokens_grant ON oauth_tokens (grant_id);
CREATE INDEX IF NOT EXISTS oauth_tokens_user ON oauth_tokens (user_id);
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class IdentityError(Exception):
    """Google did not confirm who the user is."""


@dataclass(frozen=True)
class Identity:
    sub: str
    email: str
    name: str = ""


@dataclass(frozen=True)
class User:
    id: str
    google_sub: str
    email: str
    name: str
    created_at: float
    approved_at: float | None
    blocked_at: float | None
    last_seen_at: float | None

    @property
    def status(self) -> str:
        return "blocked" if self.blocked_at is not None else "approved" if self.approved_at is not None else "pending"


class Google:
    """The Google leg: where to send the user, and who came back. Uses the authorization-code flow; the identity
    is read from Google's userinfo endpoint with the token Google just issued over TLS, so no signature of ours
    is involved. Only a verified email address counts."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, post: Callable | None = None,
                 get: Callable | None = None):
        self.client_id, self._secret, self.redirect_uri, self._post, self._get = client_id, client_secret, redirect_uri, post, get

    def url(self, state: str) -> str:
        return GOOGLE_AUTH_URL + "?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.redirect_uri, "response_type": "code",
            "scope": "openid email profile", "state": state, "prompt": "select_account"})

    def identify(self, code: str) -> Identity:
        import httpx

        post, get = self._post or httpx.post, self._get or httpx.get
        try:
            r = post(GOOGLE_TOKEN_URL, data={"code": code, "client_id": self.client_id, "client_secret": self._secret,
                                             "redirect_uri": self.redirect_uri, "grant_type": "authorization_code"}, timeout=15)
            token = r.json().get("access_token") if r.status_code == 200 else None
            if not token:
                raise IdentityError("no token")
            r = get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            info = r.json() if r.status_code == 200 else {}
        except IdentityError:
            raise
        except Exception as exc:
            raise IdentityError(type(exc).__name__) from None
        sub, email = str(info.get("sub") or ""), str(info.get("email") or "").strip().lower()
        if not sub or not email or info.get("email_verified") not in (True, "true"):
            raise IdentityError("no verified email")
        return Identity(sub[:255], email[:254], " ".join(str(info.get("name") or "").split())[:120])


_USER_COLUMNS = "id, google_sub, email, name, created_at, approved_at, blocked_at, last_seen_at"


class Users:
    def __init__(self, db_path: Path | str, clock: Callable[[], float] = time.time, auto_approve: tuple[str, ...] = ()):
        self.db_path, self.clock = Path(db_path), clock
        self.auto_approve = {e.strip().lower() for e in auto_approve if e.strip()}
        self._seen: dict[str, float] = {}
        init_schema(self.db_path, SCHEMA)

    def sign_in(self, who: Identity) -> tuple[User, bool]:
        """The user of this Google account and whether this sign-in created them (a signup). The email and name
        follow the Google account; an address in ``auto_approve`` is approved on the spot."""
        now = self.clock()
        with transaction(self.db_path) as con:
            row = con.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?", (who.sub,)).fetchone()
            new = row is None
            if new:
                for _ in range(5):
                    user_id = "u_" + secrets.token_hex(4)
                    if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
                        break
                con.execute("INSERT INTO users (id, google_sub, email, name, created_at) VALUES (?, ?, ?, ?, ?)",
                            (user_id, who.sub, who.email, who.name, now))
            else:
                con.execute("UPDATE users SET email = ?, name = ? WHERE google_sub = ?", (who.email, who.name, who.sub))
            if who.email in self.auto_approve:
                con.execute("UPDATE users SET approved_at = ? WHERE google_sub = ? AND approved_at IS NULL", (now, who.sub))
            return User(*con.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE google_sub = ?", (who.sub,)).fetchone()), new

    def _query(self, where: str = "", args: tuple = ()) -> list[User]:
        con = connect(self.db_path)
        try:
            return [User(*r) for r in con.execute(f"SELECT {_USER_COLUMNS} FROM users {where} ORDER BY created_at, id", args)]
        finally:
            con.close()

    def get(self, user_id: str) -> User | None:
        found = self._query("WHERE id = ?", (user_id,))
        return found[0] if found else None

    def find(self, who: str) -> list[User]:
        """Users with this id or this email address (an address can belong to more than one Google account)."""
        return self._query("WHERE id = ? OR email = ?", (who.strip(), who.strip().lower()))

    def all(self) -> list[User]:
        return self._query()

    def _set(self, user_id: str, sql: str) -> bool:
        with transaction(self.db_path) as con:
            return con.execute(f"UPDATE users SET {sql} WHERE id = ?", (self.clock(), user_id)).rowcount == 1

    def approve(self, user_id: str) -> bool:
        """Let the user in (also lifts a block). False for an unknown id."""
        return self._set(user_id, "blocked_at = NULL, approved_at = COALESCE(approved_at, ?)")

    def block(self, user_id: str) -> bool:
        """Shut the user out and delete their tokens. False for an unknown id."""
        done = self._set(user_id, "blocked_at = ?")
        with transaction(self.db_path) as con:
            con.execute("DELETE FROM oauth_tokens WHERE user_id = ?", (user_id,))
            con.execute("DELETE FROM oauth_codes WHERE user_id = ?", (user_id,))
        return done

    def seen(self, user_id: str) -> None:
        """Note that the user made a request (written at most once a minute per user)."""
        now = self.clock()
        if now - self._seen.get(user_id, 0.0) < SEEN_EVERY_S:
            return
        if len(self._seen) > 10_000:
            self._seen.clear()
        self._seen[user_id] = now
        with transaction(self.db_path) as con:
            con.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, user_id))
