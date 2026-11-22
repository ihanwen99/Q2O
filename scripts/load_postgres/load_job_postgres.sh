#!/bin/bash
# Copyright 2022 The Balsa Authors.
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

set -ex

DATA_DIR="$(cd "$1" && pwd)"
DBNAME=${2:-imdbload}

createdb $DBNAME

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
psql $DBNAME -f "$DIR/schema.sql"
psql $DBNAME -f "$DIR/fkindexes.sql"

psql $DBNAME -c "\copy name from '$DATA_DIR/name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy char_name from '$DATA_DIR/char_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy comp_cast_type from '$DATA_DIR/comp_cast_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy company_name from '$DATA_DIR/company_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy company_type from '$DATA_DIR/company_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy info_type from '$DATA_DIR/info_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy keyword from '$DATA_DIR/keyword.csv' escape '\' csv header" &
psql $DBNAME -c "\copy kind_type from '$DATA_DIR/kind_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy link_type from '$DATA_DIR/link_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy role_type from '$DATA_DIR/role_type.csv' escape '\' csv header" &
psql $DBNAME -c "\copy title from '$DATA_DIR/title.csv' escape '\' csv header" &
psql $DBNAME -c "\copy aka_name from '$DATA_DIR/aka_name.csv' escape '\' csv header" &
psql $DBNAME -c "\copy aka_title from '$DATA_DIR/aka_title.csv' escape '\' csv header" &
psql $DBNAME -c "\copy cast_info from '$DATA_DIR/cast_info.csv' escape '\' csv header" &
psql $DBNAME -c "\copy complete_cast from '$DATA_DIR/complete_cast.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_companies from '$DATA_DIR/movie_companies.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_info from '$DATA_DIR/movie_info.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_info_idx from '$DATA_DIR/movie_info_idx.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_keyword from '$DATA_DIR/movie_keyword.csv' escape '\' csv header" &
psql $DBNAME -c "\copy movie_link from '$DATA_DIR/movie_link.csv' escape '\' csv header" &
psql $DBNAME -c "\copy person_info from '$DATA_DIR/person_info.csv' escape '\' csv header" &
wait

psql $DBNAME -f "$DIR/add_fks.sql"

psql $DBNAME -c "ANALYZE VERBOSE;"
