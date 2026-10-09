#!/usr/bin/env python3
"""Submit solutions to Library Checker as the account owner. Runs on the Mac only.

  submit.py login           Ask for email and password once (hidden). Only a renewable
                            token is kept, in macOS Keychain; the password is not stored.
  submit.py PROBLEM FILE    Submit FILE as C++23, wait for the verdict, print it.

At most CAP submissions per version: earlier submissions with the same source count.
"""
from __future__ import annotations  # macOS ships Python 3.9

import getpass
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://v3.api.judge.yosupo.jp'
FIREBASE_KEY = 'AIzaSyCmpkoMVbKRDm2H0MJHB0iZ43uQtSqiLV0'  # public web key of judge.yosupo.jp
USER = 'Aiyiyi'
CAP = 5
KEYCHAIN = ['-a', 'lc-opt', '-s', 'lc-opt-judge-token']
FINAL = {'AC', 'WA', 'RE', 'TLE', 'MLE', 'PE', 'Fail', 'CE', 'IE', 'ICE'}


def call(url: str, data: dict | None = None, token: str | None = None, form: bool = False) -> dict:
    headers = {'Referer': 'https://judge.yosupo.jp/'}
    body = None
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode()
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        else:
            body = json.dumps(data).encode()
            headers['Content-Type'] = 'application/json'
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        sys.exit(f'{url.split("?")[0]}: HTTP {e.code} {e.read().decode()[:300]}')


def login() -> int:
    email = input('Library Checker email: ')
    password = getpass.getpass('Password (hidden): ')
    auth = call(f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_KEY}',
                {'email': email, 'password': password, 'returnSecureToken': True})
    name = call(f'{API}/auth/current_user', token=auth['idToken'])['user']['name']
    if name != USER:
        sys.exit(f'logged in as {name}, expected {USER}; nothing stored')
    subprocess.run(['security', 'add-generic-password', '-U', *KEYCHAIN, '-w', auth['refreshToken']], check=True)
    print(f'Logged in as {name}. Token stored in Keychain.')
    return 0


def id_token() -> str:
    """Exchange the stored refresh token for a one-hour ID token."""
    stored = subprocess.run(['security', 'find-generic-password', *KEYCHAIN, '-w'], capture_output=True, text=True)
    if stored.returncode != 0:
        sys.exit('not logged in: run `python3 tools/submit.py login` in a terminal')
    refreshed = call(f'https://securetoken.googleapis.com/v1/token?key={FIREBASE_KEY}',
                     {'grant_type': 'refresh_token', 'refresh_token': stored.stdout.strip()}, form=True)
    return refreshed['id_token']


def used(problem: str, source: str) -> int:
    query = urllib.parse.urlencode({'problem': problem, 'user': USER, 'limit': 1000})
    submissions = call(f'{API}/submissions?{query}')['submissions']
    return sum(call(f'{API}/submissions/{s["id"]}')['source'] == source for s in submissions)


def submit(problem: str, file: str) -> int:
    source = Path(file).read_text()
    count = used(problem, source)
    if count >= CAP:
        sys.exit(f'{problem}: cap reached ({count}/{CAP} submissions of this version)')
    sid = call(f'{API}/submit', {'problem': problem, 'source': source, 'lang': 'cpp'}, token=id_token())['id']
    print(f'submission {sid} ({count + 1}/{CAP}): https://judge.yosupo.jp/submission/{sid}', flush=True)
    for _ in range(120):
        time.sleep(5)
        result = call(f'{API}/submissions/{sid}')['overview']
        if result['status'] in FINAL:
            print(f'{result["status"]} {result["time"] * 1000:.0f} ms {result["memory"] / 2**20:.1f} MiB')
            return 0 if result['status'] == 'AC' else 1
    sys.exit(f'submission {sid}: no verdict after 10 minutes')


if __name__ == '__main__':
    if sys.argv[1:] == ['login']:
        sys.exit(login())
    if len(sys.argv) == 3:
        sys.exit(submit(sys.argv[1], sys.argv[2]))
    sys.exit(__doc__)
