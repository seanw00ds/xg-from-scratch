"""Pull shot events from the StatsBomb open data repo and write one flat CSV.

Population: men's senior international tournaments, 2018-2024.
Penalties are dropped (fixed situation, not a normal shot).
"""

import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

COMPS = [
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
    (55, 43, "Euro 2020"),
    (55, 282, "Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]

FIELDS = [
    "match_id", "competition", "player", "team", "minute", "period",
    "x", "y", "body_part", "technique", "shot_type", "play_pattern",
    "first_time", "under_pressure", "open_goal", "follows_dribble",
    "n_opponents_in_frame", "defenders_in_cone", "keeper_x", "keeper_y",
    "assist_type", "assist_height",
    "statsbomb_xg", "outcome", "goal",
]


def fetch(url, attempts=4):
    """Retry on failure. Without this a transient SSL timeout silently drops a
    match, the dataset quietly changes size between runs, and the numbers in
    the report stop matching the numbers anyone else gets."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def pass_info(events):
    """id -> (pass type, pass height) so a shot can look up its assist."""
    out = {}
    for e in events:
        if e.get("type", {}).get("name") == "Pass":
            p = e["pass"]
            kind = p.get("type", {}).get("name", "Regular")
            if p.get("cross"):
                kind = "Cross"
            elif p.get("through_ball"):
                kind = "Through Ball"
            elif p.get("cut_back"):
                kind = "Cut Back"
            out[e["id"]] = (kind, p.get("height", {}).get("name", "Unknown"))
    return out


POST_A = (120.0, 36.0)
POST_B = (120.0, 44.0)


def in_triangle(p, a, b, c):
    """Point-in-triangle by the sign of three cross products."""
    def cross(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    has_neg = min(d1, d2, d3) < 0
    has_pos = max(d1, d2, d3) > 0
    return not (has_neg and has_pos)


def freeze_frame_counts(shot, loc):
    """Who is in the way: opponents in frame, defenders inside the shooting
    cone (the triangle from the ball to each post), and the keeper's position."""
    ff = shot.get("freeze_frame")
    if not ff:
        return "", "", "", ""
    opponents = [p for p in ff if not p["teammate"]]
    keeper = next(
        (p for p in ff if p.get("position", {}).get("name") == "Goalkeeper"
         and not p["teammate"]),
        None,
    )
    kx, ky = (keeper["location"] if keeper else ("", ""))
    in_cone = sum(
        1 for p in opponents
        if p is not keeper and in_triangle(p["location"], loc, POST_A, POST_B)
    )
    return len(opponents), in_cone, kx, ky


def shots_from_match(match):
    mid = match["match_id"]
    try:
        events = fetch(f"{BASE}/events/{mid}.json")
    except Exception as exc:
        # Loud, not silent: a dropped match changes every number downstream.
        print(f"  FAILED {mid} after retries: {exc}", file=sys.stderr)
        raise

    passes = pass_info(events)
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        s = e["shot"]
        shot_type = s.get("type", {}).get("name", "")
        if shot_type == "Penalty":
            continue
        n_opp, in_cone, kx, ky = freeze_frame_counts(s, e["location"])
        assist = passes.get(s.get("key_pass_id"), ("None", "None"))
        outcome = s.get("outcome", {}).get("name", "")
        rows.append({
            "match_id": mid,
            "competition": match["_label"],
            "player": e.get("player", {}).get("name", ""),
            "team": e.get("team", {}).get("name", ""),
            "minute": e.get("minute", ""),
            "period": e.get("period", ""),
            "x": e["location"][0],
            "y": e["location"][1],
            "body_part": s.get("body_part", {}).get("name", ""),
            "technique": s.get("technique", {}).get("name", ""),
            "shot_type": shot_type,
            "play_pattern": e.get("play_pattern", {}).get("name", ""),
            "first_time": int(bool(s.get("first_time"))),
            "under_pressure": int(bool(e.get("under_pressure"))),
            "open_goal": int(bool(s.get("open_goal"))),
            "follows_dribble": int(bool(s.get("follows_dribble"))),
            "n_opponents_in_frame": n_opp,
            "defenders_in_cone": in_cone,
            "keeper_x": kx,
            "keeper_y": ky,
            "assist_type": assist[0],
            "assist_height": assist[1],
            "statsbomb_xg": s.get("statsbomb_xg", ""),
            "outcome": outcome,
            "goal": int(outcome == "Goal"),
        })
    return rows


def main(out_path):
    matches = []
    for comp_id, season_id, label in COMPS:
        got = fetch(f"{BASE}/matches/{comp_id}/{season_id}.json")
        for m in got:
            m["_label"] = label
        matches += got
        print(f"{label}: {len(got)} matches")

    print(f"downloading events for {len(matches)} matches ...")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for batch in pool.map(shots_from_match, matches):
            rows += batch

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    goals = sum(r["goal"] for r in rows)
    n_matches = len({r["match_id"] for r in rows})
    print(f"{len(rows)} shots, {goals} goals ({goals / len(rows):.1%}), "
          f"{n_matches} matches -> {out_path}")
    assert n_matches == len(matches), (
        f"expected {len(matches)} matches, got {n_matches}: a download was lost")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/shots.csv")
