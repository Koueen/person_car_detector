import json
import tyro
import requests
from PIL import Image, ImageDraw

def request_prediction(url: str, img_path: str) -> str:
    with open(img_path, 'rb') as file:
        response = requests.post(url, files={'data': file})
    print("Status Code:", response.status_code)
    print("Response Text:\n", response.text)
    return response.text


def draw_bboxes(text_response: str, img_path: str):
    result = json.loads(text_response)
    img = Image.open(img_path)
    color_map = {'Person': (56, 41, 131), 'Car': (196, 166, 44)}
    width = 2
    draw = ImageDraw.Draw(img)
    for i, detection in enumerate(result['RESULTS']):
        draw.rectangle(
            detection[f'detection_{i}']['bbox'], outline=color_map[detection[f'detection_{i}']['label']], width=width
        )
    img.save('app/images/output/image_with_bbox.jpg')


def main(url: str ="http://localhost:8080/predictions/fcos_model", img_path: str = "app/images/person_car_2.jpg"):
    text_response = request_prediction(url, img_path)
    draw_bboxes(text_response, img_path)


if __name__ == '__main__':
    tyro.cli(main)
