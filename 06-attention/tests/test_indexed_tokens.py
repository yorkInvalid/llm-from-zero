import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexed_tokens import TokenWindowWriter, IndexedTokenWindows, dataset_counts, split_fingerprint
from text_pretraining import make_windows
from prepare_industry_100m import HoldoutGuard


class IndexedTokenTests(unittest.TestCase):
    def test_windows_equal_existing_pipeline_and_preserve_document_boundaries(self):
        tok=Tokenizer(models.WordLevel({'<PAD>':0,'<BOS>':1,'<EOS>':2,'a':3,'b':4,'<UNK>':5},unk_token='<UNK>'))
        tok.pre_tokenizer=pre_tokenizers.Whitespace()
        docs=['a b a b a b a', 'b', 'a b a b']
        expected=make_windows(docs,tok,4)
        with tempfile.TemporaryDirectory() as d:
            writer=TokenWindowWriter(d,context_length=4)
            for doc in docs:writer.add_document([1,*tok.encode(doc).ids,2])
            writer.finish()
            actual=IndexedTokenWindows(d,'train',4)
            self.assertEqual(dataset_counts(actual),dataset_counts(expected))
            for (x,y),(a,b) in zip(expected,actual):
                torch.testing.assert_close(x,a);torch.testing.assert_close(y,b)
            self.assertEqual(len(actual[:2]),2)
            with self.assertRaisesRegex(ValueError,'context'):IndexedTokenWindows(d,'train',8)

    def test_exact_budget_truncates_without_fabricating_eos(self):
        with tempfile.TemporaryDirectory() as d:
            writer=TokenWindowWriter(d,context_length=2)
            self.assertEqual(writer.add_document([1,3,4,5,6,2],max_targets=3),3)
            writer.finish()
            rows=IndexedTokenWindows(d,'train',2)
            self.assertEqual(dataset_counts(rows)['tokens'],3)
            self.assertEqual(torch.cat([y for _,y in rows]).tolist(),[3,4,5])
            before=split_fingerprint(d,'train')
            p=Path(d)/'train.tokens.u32'
            with p.open('r+b') as f:
                f.seek(4);f.write((7).to_bytes(4,'little'))
            self.assertNotEqual(before,split_fingerprint(d,'train'))

    def test_holdout_overlap_at_every_alignment_is_blocked(self):
        shared=''.join(chr(0x4e00+i) for i in range(100))
        guard=HoldoutGuard(['保留文本'+shared+'验证后缀'])
        for shift in range(50):
            self.assertTrue(guard.overlaps('x'*shift+shared+'新的后缀'),shift)
        self.assertFalse(guard.overlaps('不同的文字'*40))


if __name__=='__main__':unittest.main()
