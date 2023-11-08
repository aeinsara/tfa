import os
import io
import json
import copy
import torch
import logging
import itertools
import contextlib
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.structures import BoxMode
from detectron2.utils import comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table
from detectron2.data.datasets.coco import convert_to_coco_json
from fsdet.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog


class DOTAEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, output_dir=None):

        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

#         DatasetCatalog.register(dataset_name, dota_function)  ////////////////////// dota
        if dataset_name in DatasetCatalog.list(): # ......sara added beacuse of error "Dataset '{}' is already registered!"
            DatasetCatalog.remove(dataset_name)
#         print("dataset_name ::::::::::::"+str(dataset_name))
#         /home/aeen/fewshot/Datasets/dota/dotasplit/datasplit/
        register_coco_instances(dataset_name, {}, "/home/aeen/fewshot/Datasets/dota/dotasplit/datasplit/5k.json", "/home/aeen/fewshot/Datasets/dota/test1024")
        
        self._metadata = MetadataCatalog.get(dataset_name)
#         print("dataset_name ::::::::::::"+str(self._metadata ))
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")
            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._base_classes = [
            1,2,3,4,5,6,7,8,9,10,11,12 #............... add 13,14,15,16
        ]
        self._novel_classes = [13,14,15,16]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._dota_api = COCO(json_file)
        self._do_evaluation = "annotations" in self._dota_api.dataset

    def reset(self):
        self._predictions = []
        self._dota_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a DOTA model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a DOTA model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[DOTAEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for DOTA format ...")
        self._dota_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for DOTA
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._dota_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "dota_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._dota_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                    ("all", None, self._metadata.get("thing_classes")),
                    ("base", self._base_classes, self._metadata.get("base_classes")),
                    ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                dota_eval = (
                    _evaluate_predictions_on_dota(
                        self._dota_api, self._dota_results, "bbox", classes,
                    )
                    if len(self._dota_results) > 0
                    else None  # dota_api does not handle empty results very well
                )
                res_ = self._derive_dota_results(dota_eval, "bbox", class_names=names)
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b"+metric] = res_[metric]
                        elif split == "novel":
                            res["n"+metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            dota_eval = (
                _evaluate_predictions_on_dota(
                    self._dota_api, self._dota_results, "bbox",
                )
                if len(self._dota_results) > 0
                else None  # dota api does not handle empty results very well
            )
            res = self._derive_dota_results(
                dota_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_dota_results(self, dota_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            dota_eval (None or COCOeval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if dota_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(dota_eval.stats[idx] * 100) \
                for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
                create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        precisions = dota_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_dota(dota_gt, dota_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(dota_results) > 0

    dota_dt = dota_gt.loadRes(dota_results)
    dota_eval = COCOeval(dota_gt, dota_dt, iou_type)
    if catIds is not None:
        dota_eval.params.catIds = catIds
    dota_eval.evaluate()
    dota_eval.accumulate()
    dota_eval.summarize()

    return dota_eval

# def dota_function():

#   return list[dict] in the following format
