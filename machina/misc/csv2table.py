# Copyright 2018 DeepX Inc. All Rights Reserved.
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

import csv
import os
import terminaltables


def csv2table(filename, save_dir=None, output_filename='scores-table.txt'):
    with open(filename, newline='') as f:
        data = list(csv.reader(f))
        table = terminaltables.AsciiTable(data)
        table.inner_row_border = False
        table.CHAR_H_INNER_HORIZONTAL = '-'
        table.CHAR_OUTER_TOP_LEFT = ''
        table.CHAR_OUTER_TOP_RIGHT = ''
        table.CHAR_OUTER_TOP_HORIZONTAL = ''
        table.CHAR_OUTER_TOP_INTERSECT = ''
        table.CHAR_OUTER_BOTTOM_LEFT = '|'
        table.CHAR_OUTER_BOTTOM_RIGHT = '|'
        table.CHAR_OUTER_BOTTOM_HORIZONTAL = ' '
        table.CHAR_OUTER_BOTTOM_INTERSECT = '|'
        table.CHAR_H_OUTER_LEFT_INTERSECT = '|'
        table.CHAR_H_OUTER_RIGHT_INTERSECT = '|'
        output = table.table.lstrip().rstrip()

    if save_dir is None:
        save_dir = os.path.dirname(filename)
    with open(os.path.join(save_dir, output_filename), 'w') as f:
        f.write(output)
