__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.spice_object_categories(scorers)
                
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
        
    # new   
    def spice_object_categories(self, scorers):
        
        # SPICE scorer
        spice_scorer = None
        for scorer, method in scorers:
            if method == 'SPICE':
                # Objects are extracted from SPICE semantic tuples by collecting all single-node tuples (noun nodes) 
                # from the generated and reference scene graphs.
                imgId_to_objects = {}
                for item in scorer.results:
                    imgId_to_objects[item['image_id']] = {
                    'pred': item.get('test_tuples', []),
                    'gt': item.get('ref_tuples', []),
                }
            
                # imgToEval
                for imgId, eva in self.imgToEval.items():
                    if imgId in imgId_to_objects:
                        eva.setdefault(method, {})
                        eva[method]['ObjectCategories'] = imgId_to_objects[imgId]
                        eva[method]['ObjectAnalysis'] =self.analyze_spice_objects(imgId_to_objects[imgId])
                        
                        
    def analyze_spice_objects(self, obj_cats):
        """
        Analyze SPICE object-level errors.
    
        Returns:
            {
              "correct": [...],
              "hallucinated": [...],
              "missing": [...],
              "precision": float,
              "recall": float,
              "hallucination_rate": float
            }
        """
    
        pred_items = obj_cats.get("pred", [])
        gt_items = obj_cats.get("gt", [])
    
        # -------- extract object names --------
        # pred objects (all)
        pred_objects = {
            t["tuple"][0]
            for t in pred_items
            if len(t.get("tuple", [])) == 1
        }
    
        # gt objects (all)
        gt_objects = {
            t["tuple"][0]
            for t in gt_items
            if len(t.get("tuple", [])) == 1
        }
    
        # correct objects (pred กษ gt)
        # correct_objects = pred_objects & gt_objects
        
        correct_objects = {
            t["tuple"][0]
            for t in pred_items
            if len(t.get("tuple", [])) == 1 and t['truth_value'] is True
        }
    
        # hallucinated: predicted but not in gt
        # hallucinated_objects = pred_objects - gt_objects
        
        hallucinated_objects = {
            t["tuple"][0]
            for t in pred_items
            if len(t.get("tuple", [])) == 1 and t['truth_value'] is False
        }
    
        # missing: in gt but not predicted
        # missing_objects = gt_objects - pred_objects
        
        missing_objects = {
            t["tuple"][0]
            for t in gt_items
            if len(t.get("tuple", [])) == 1 and t['truth_value'] is False
        }
        
        
    
        # -------- metrics --------
        num_pred = len(pred_objects)
        num_gt = len(gt_objects)
        num_correct = len(correct_objects)
        num_hallu = len(hallucinated_objects)
    
        precision = num_correct / num_pred if num_pred > 0 else 0.0
        recall = num_correct / num_gt if num_gt > 0 else 0.0
        hallucination_rate = num_hallu / num_pred if num_pred > 0 else 0.0

        return {
            "pred_objects": sorted(pred_objects),
            "gt_objects": sorted(gt_objects),
            "correct": sorted(correct_objects),
            "hallucinated": sorted(hallucinated_objects),
            "missing": sorted(missing_objects),
            "precision": precision,
            "recall": recall,
            "hallucination_rate": hallucination_rate
        }
