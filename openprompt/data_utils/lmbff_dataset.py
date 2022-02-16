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

import os
from typing import List
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
import json, csv

# logger = log.get_logger(__name__)

class SSTDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
    
    def get_examples(self, data_dir, split):
        #print('split:', split)
        if False:
        #if split == 'test':
            input_dir = data_dir
            examples = []
            input_dir2 = os.path.join(input_dir,split)
            for i, fname in enumerate(os.listdir(input_dir2)):
                label = int(fname[-5])
                text_a = open(os.path.join(input_dir2,fname),encoding="utf-8").read()
                guid = f"{split}_{i}"
                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)
            return examples
        else:
            path = os.path.join(data_dir, f"{split}.tsv")
            examples = []
            with open(path, encoding='utf-8')as f:
                lines = f.readlines()
                for idx, line in enumerate(lines[1:]):
                    linelist = line.strip().split('\t')
                    text_a = linelist[0]
               #print('linelist[0]:', linelist[0])
                #print('linelist[1]:', linelist[1])
                    if not linelist[1].isdigit():
                        continue
                    label = int(linelist[1])                
                    guid = "%s-%s" % (split, idx)
                    example = InputExample(guid=guid, text_a=text_a, label=label)
                    examples.append(example)
            return examples


"""
class SSTDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
    
    def get_examples(self, data_dir, split):
        #print('split:', split)
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
               #print('linelist[0]:', linelist[0])
                #print('linelist[1]:', linelist[1])
                if not linelist[1].isdigit():
                    continue
                label = int(linelist[1])                
                guid = "%s-%s" % (split, idx)
                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)
        return examples

class SSTDataProcessor(DataProcessor):
    #TODO test needed
    def __init__(self):
        #raise NotImplementedError
        super().__init__()
        self.labels = ["negative", "positive"]

    def get_examples(self, data_dir, split):
        examples = []
        #print('111')
        path = os.path.join(data_dir,"{}.tsv".format(split))
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                text_a = example_json['sentence'].strip()
                if not example_json['label'].isdigit():
                    continue
                example = InputExample(guid=str(idx), text_a=text_a, label=int(example_json['label']))
                examples.append(example)
        return examples
"""

class SNLIDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ['entailment', 'neutral', 'contradiction']
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                guid = "%s-%s" % (split, idx)
                label = linelist[-1]
                text_a = linelist[7]
                text_b = linelist[8]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
                examples.append(example)
        return examples

PROCESSORS = {
    "sst-2": SSTDataProcessor,
    "snli": SNLIDataProcessor
}
