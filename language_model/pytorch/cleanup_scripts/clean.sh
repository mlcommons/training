#!/bin/bash
# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================

filein=$1
fileout=$2

echo "Further clean up: $filein => $fileout"

cmd="cat $filein "
cmd+="| grep -v '^<doc [^>]*>$' "
cmd+="| grep -vE '\[\[Category:[^][]*\]\]' "
cmd+="| sed 's/\[\[\([^]|[]*\)\]\]/\1/g' "
cmd+="| sed 's/\[\[\([^]|[]*\)\]\]/\1/g' "
cmd+="| sed 's/\[\[[^]|[]*|\([^]|[]*\)\]\]/\1/g' "
cmd+="| sed 's/\[\[[^]|[]*|\([^]|[]*\)\]\]/\1/g' "
cmd+="| sed 's/\[\[[:]*[Ff]ile:[^][]*\]\]//g' "
cmd+="| sed 's/\[\[[Mm]edia:[^][]*\]\]//g' "
cmd+="| sed 's/\[\[[Ii]mage:[^][]*\]\]//g' "
cmd+="| sed 's/\[\([^]|[]*\)\]/\1/g' "
cmd+="| sed 's/\[\[\([^][]*\)\]\]//g' "
cmd+="| sed 's/alt=//g' "
cmd+="| sed 's/<\/doc>/\r/g' "
cmd+="| sed 's/<chem\([^<]*\)<\/chem>/\1/g' "
cmd+="| sed 's/<ins\([^<]*\)<\/ins>/\1/g' "
cmd+="| sed 's/<\, ref \([^<]*\)<\/ref>//g' "
cmd+="| sed 's/<includeonly\([^<]*\)<\/includeonly>//g' "
cmd+="| sed 's/<graph\([^<]*\)<\/graph>//g' "
cmd+="| sed 's/<section\([^\\]*\)\/>//g' "
cmd+="| sed 's/<meta\([^\\]*\)\/>//g' "
cmd+="| sed 's/<hr\([^\\]*\)\/>//g' "
cmd+="| sed 's/<gallery\([^>]*\)>//g' "
cmd+="| sed 's/<ref\([^<]*\)<\/ref>//g' "
cmd+="| sed 's/<ref\([^>]*\)>//g' "
cmd+="| sed 's/<http\([^>]*\)>//g' "
cmd+="| sed 's/<Ref\([^>]*\)>//g' "
cmd+="| sed 's/<mapframe \([^\/]*\)\/>//g' "
cmd+="| sed 's/<mapframe\([^>]*\)>//g' "
cmd+="| sed 's/<\/mapframe>//g' "
cmd+="| sed 's/<poem>//g' "
cmd+="| sed 's/<\/poem>//g' "
cmd+="| sed 's/<math>//g' "
cmd+="| sed 's/<\/math>//g' "
cmd+="| sed 's/<ref>//g' "
cmd+="| sed 's/<\/ref>//g' "
cmd+="| sed 's/<div\([^>]*\)>//g' "
cmd+="| sed 's/<\/div\([^>]*\)>//g' "
cmd+="| sed 's/<\/div style>//g' "
cmd+="| sed 's/<\/div>//g' "
cmd+="| sed 's/<sup>//g' "
cmd+="| sed 's/<\/sup>//g' "
cmd+="| sed 's/<br>//g' "
cmd+="| sed 's/<\/br>//g' "
cmd+="| sed 's/<BR>//g' "
cmd+="| sed 's/<\/BR>//g' "
cmd+="| sed 's/<Br>//g' "
cmd+="| sed 's/<\/Br>//g' "
cmd+="| sed 's/<del>//g' "
cmd+="| sed 's/<\/del>//g' "
cmd+="| sed 's/<nowiki>//g' "
cmd+="| sed 's/<\/nowiki>//g' "
cmd+="| sed 's/<NOWIKI>//g' "
cmd+="| sed 's/<\/NOWIKI>//g' "
cmd+="| sed 's/<onlyinclude>//g' "
cmd+="| sed 's/<\/onlyinclude>//g' "
cmd+="| sed 's/<includeonly>//g' "
cmd+="| sed 's/<\/includeonly>//g' "
cmd+="| sed 's/<small>//g' "
cmd+="| sed 's/<\/small>//g' "
cmd+="| sed 's/<chem>//g' "
cmd+="| sed 's/<\/chem>//g' "
cmd+="| sed 's/<noinclude>//g' "
cmd+="| sed 's/<\/noinclude>//g' "
cmd+="| sed 's/<gallery>//g' "
cmd+="| sed 's/<\/gallery>//g' "
cmd+="| sed 's/<graph>{//g' "
cmd+="| sed 's/<graph>//g' "
cmd+="| sed 's/}<\/graph>//g' "
cmd+="| sed 's/<\/graph>//g' "
cmd+="| sed 's/<\/references>//g' "
cmd+="| sed 's/<poem \([^>]*\)>//g' "
# cmd+="| grep -v '^[ \t]*$' "
cmd+="> $fileout"

bash -c "${cmd[@]}"
