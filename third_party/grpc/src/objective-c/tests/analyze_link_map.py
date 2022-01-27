#!/usr/bin/python
# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script analyzes link map file generated by Xcode. It calculates and
# prints out the sizes of each dependent library and the total sizes of the
# symbols.
# The script takes one parameter, which is the path to the link map file.

import sys
import re

table_tag = {}
state = "start"

table_stats_symbol = {}
table_stats_dead = {}
section_total_size = 0
symbol_total_size = 0


file_import = sys.argv[1]
lines = list(open(file_import))
for line in lines:
    line_stripped = line[:-1]
    if "# Object files:" == line_stripped:
        state = "object"
        continue
    elif "# Sections:" == line_stripped:
        state = "section"
        continue
    elif "# Symbols:" == line_stripped:
        state = "symbol"
        continue
    elif "# Dead Stripped Symbols:" == line_stripped:
        state = "dead"
        continue

    if state == "object":
        segs = re.search("(\[ *[0-9]*\]) (.*)", line_stripped)
        table_tag[segs.group(1)] = segs.group(2)

    if state == "section":
        if len(line_stripped) == 0 or line_stripped[0] == "#":
            continue
        segs = re.search("^(.+?)\s+(.+?)\s+.*", line_stripped)
        section_total_size += int(segs.group(2), 16)

    if state == "symbol":
        if len(line_stripped) == 0 or line_stripped[0] == "#":
            continue
        segs = re.search("^.+?\s+(.+?)\s+(\[.+?\]).*", line_stripped)
        target = table_tag[segs.group(2)]
        target_stripped = re.search("^(.*?)(\(.+?\))?$", target).group(1)
        size = int(segs.group(1), 16)
        if not target_stripped in table_stats_symbol:
            table_stats_symbol[target_stripped] = 0
        table_stats_symbol[target_stripped] += size

print("Sections total size: %d" % section_total_size)

for target in table_stats_symbol:
    print(target)
    print(table_stats_symbol[target])
    symbol_total_size += table_stats_symbol[target]

print("Symbols total size: %d" % symbol_total_size)
