import io
import json
from typing import List

import torch
from PIL import Image
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.detection_thr = 0.5
        self.label_mapping = {1: 'Person', 3: 'Car'}

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.model = torch.load(f'{model_dir}/fcos_model.pt')
        self.model.eval()

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        image = Image.open(io.BytesIO(preprocessed_data))
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        preprocess = weights.transforms()
        batch = [preprocess(image)]
        return batch

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(model_input)[0]
        return model_output

    def postprocess(self, prediction) -> List[dict]:
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        mask_score = prediction['scores'] > self.detection_thr
        mask_labels = (prediction['labels'] == 1) | (prediction['labels'] == 3)
        final_mask = mask_score & mask_labels
        indexes_preds = torch.nonzero(final_mask)
        results_bbox = []
        for i, i_pr in enumerate(indexes_preds):
            bbox = [round(coord, 3) for coord in prediction['boxes'][i_pr][0].tolist()]
            results_bbox.append(
                {
                    f'detection_{i}': {
                        'bbox': bbox,
                        'label': self.label_mapping[prediction['labels'][i_pr][0].item()],
                        'confidence': round(prediction['scores'][i_pr][0].item(), 3),
                    }
                }
            )
        return results_bbox

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        json_result = self.postprocess(model_output)
        return [{'RESULTS': json_result}]
