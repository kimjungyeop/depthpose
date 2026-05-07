#!/usr/bin/env python3
"""Dedupe pandoc footnotes: when the same key is cited N times, pandoc 3.x emits
N separate <li> entries with identical content. Collapse them to one canonical
entry and rewrite all body refs + the displayed superscript numbers."""
import sys
from pathlib import Path
from bs4 import BeautifulSoup

path = Path(sys.argv[1] if len(sys.argv) > 1 else "index.html")
soup = BeautifulSoup(path.read_text(), "html.parser")

aside = soup.find("aside", id="footnotes")
ol = aside.find("ol")
items = ol.find_all("li", recursive=False)

# Phase 1: identify canonical IDs by content
seen = {}
remap = {}
for li in items:
    p = li.find("p")
    back = p.find("a", class_="footnote-back")
    back_extracted = back.extract() if back else None
    text = p.get_text().strip()
    if back_extracted:
        p.append(back_extracted)
    fnid = li["id"]
    if text in seen:
        remap[fnid] = seen[text]
    else:
        seen[text] = fnid

# Phase 2: rewrite body refs to point to canonical
for a in soup.find_all("a", class_="footnote-ref"):
    href = a.get("href", "")
    if href.startswith("#") and href[1:] in remap:
        a["href"] = "#" + remap[href[1:]]

# Phase 3: drop the duplicate <li>s
for li in items:
    if li["id"] in remap:
        li.decompose()

# Phase 4: renumber surviving <li>s as fn1..fnN sequentially, AND renumber
# the displayed superscript numbers in the body to match list order
canonicals = ol.find_all("li", recursive=False)
old_to_new = {li["id"]: f"fn{i+1}" for i, li in enumerate(canonicals)}

# Update the <li> ids and back-link hrefs on the way past
for li in canonicals:
    old_id = li["id"]
    li["id"] = old_to_new[old_id]
    # back-link href: pandoc emits the FIRST fnref it saw for this fn
    # Since we collapsed multiple refs onto one fn, just keep the back-link href
    # as the first body ref pointing here (the canonical fnref is the smallest-numbered one)

# Find first body ref pointing to each canonical, point back-link there
canonical_first_ref = {}
for a in soup.find_all("a", class_="footnote-ref"):
    href = a.get("href", "")
    if href.startswith("#"):
        target_old = href[1:]
        if target_old in old_to_new and target_old not in canonical_first_ref:
            canonical_first_ref[target_old] = a.get("id")

for li in canonicals:
    new_id = li["id"]
    # find the original old id by reverse-mapping
    old_id = next(o for o, n in old_to_new.items() if n == new_id)
    back = li.find("a", class_="footnote-back")
    if back and old_id in canonical_first_ref:
        back["href"] = "#" + canonical_first_ref[old_id]

# Update body refs: hrefs and the visible <sup>N</sup> number
for a in soup.find_all("a", class_="footnote-ref"):
    href = a.get("href", "")
    if href.startswith("#") and href[1:] in old_to_new:
        new_target = old_to_new[href[1:]]
        a["href"] = "#" + new_target
        sup = a.find("sup")
        if sup:
            sup.string = new_target[2:]  # strip 'fn' prefix

path.write_text(str(soup))
print(f"Deduped {len(remap)} duplicate footnote(s); {len(canonicals)} unique references remain.")
