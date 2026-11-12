"""Shared helpers: canonical JSON, hashing, Ed25519 sign/verify.

Used by the node, the prover CLI and the IVO CLI. All signatures are over
canonical JSON (sorted keys, no whitespace) of the message body *without* its
`sig` field.
"""
import base64
import hashlib
import json
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def unb64(s: str) -> bytes:
    return base64.b64decode(s)


def new_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def key_from_seed_b64(seed: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(unb64(seed))


def seed_b64(key: Ed25519PrivateKey) -> str:
    from cryptography.hazmat.primitives import serialization as s
    return b64(key.private_bytes(s.Encoding.Raw, s.PrivateFormat.Raw, s.NoEncryption()))


def pub_b64(key: Ed25519PrivateKey) -> str:
    from cryptography.hazmat.primitives import serialization as s
    return b64(key.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw))


def sign(key: Ed25519PrivateKey, body: dict) -> dict:
    """Return body + sig (body must not already contain 'sig')."""
    out = dict(body)
    out["sig"] = b64(key.sign(canon(body)))
    return out


def verify(pub: str, msg: dict) -> bool:
    body = {k: v for k, v in msg.items() if k != "sig"}
    try:
        Ed25519PublicKey.from_public_bytes(unb64(pub)).verify(unb64(msg["sig"]), canon(body))
        return True
    except (InvalidSignature, KeyError, ValueError):
        return False


def msg_hash(msg: dict) -> str:
    """Hash of a full signed message (incl. sig): the link in the hash chain."""
    return sha256_hex(canon(msg))


def load_env_file(path: str):
    """Tiny .env loader (KEY=VALUE lines), does not override existing env."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
