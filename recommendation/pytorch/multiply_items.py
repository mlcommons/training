# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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
# =============================================================================

# -*- coding: utf-8 -*-

import xlrd
import csv
from argparse import ArgumentParser
import os
import sys


def parse_args():
    parser = ArgumentParser(description="Read a MovieLens ratings.csv and add fake users"
                                        "This will be done by duplicating entries and chaging user id")
    parser.add_argument('filename', type=str,
                        help='filename of input file (should be ratings.csv')
    parser.add_argument('multiplier', type=int,
                        help='how many times to duplicate each item. Each duplicated item id will be original item id+x000000 where x=duplication number')
    return parser.parse_args()

def main():
    args = parse_args()
    input_filename=args.filename
    multiplier=args.multiplier
    csv_input_file=open(input_filename,'r')
    input_csv_size=os.path.getsize(input_filename)
    csv_reader=csv.reader(csv_input_file,delimiter=',')
    output_filename=os.path.splitext(input_filename)[0]+"_items_expanded.csv"
    output_csv_file = open(output_filename, 'w', newline='')
    csv_writer=csv.writer(output_csv_file,delimiter=',')
    header=True
    current_pos=0
    for row_index,row in enumerate(csv_reader):
        if row_index%1000==0:
            print('\r Currently in line: '+str(row_index),end='')
        row_index=row_index+1
        csv_writer.writerow(row)
        if not header:
            for i in range(1,multiplier):
                new_row=row.copy()
                original_item_id=int(row[1])
                new_item_id=original_item_id+1000000*i
                new_row[1]=str(new_item_id)
                csv_writer.writerow(new_row)
        header = False
    output_csv_file.close()
    print('')
    
if __name__ == '__main__':
    main()
