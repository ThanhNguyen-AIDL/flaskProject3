import json
from io import BytesIO

import requests
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def initial_configs():
    url = "https://designerapp.officeapps.live.com/designerapp/suggestions.ashx?"

    payload = json.dumps({
        "ImageFiles": [],
        "ImageUrls": [],
        "MediaContent": [],
        "Title": {
            "Text": ""
        },
        "SubTitle": {
            "Text": ""
        },
        "Expectations": {
            "Dimension": {
                "Width": 1080,
                "Height": 1080
            },
            "ExcludeDesignMetadata": [],
            "MaxCount": 120,
            "MinCount": 1,
            "TypeMetadata": [
                "png",
                "bmp",
                "jpg",
                "oxsd"
            ],
            "IsBrandKit": False
        },
        "Hints": {
            "Trigger": "DesignFromScratch",
            "EnableGetty3PVideos": "true",
            "EnableGetty3PVideosByHubble": "true",
            "EnablePicassoRanker": "true",
            "image2HeadingsForDFS": "true",
            "DesignQuery": "",
            "EnableMultiPageSuggestion": "true",
            "AllSizes": "true",
            "HasDalleImage": "false",
            "AllImagesAreDalleImages": "false",
            "LastDalleQuery": "",
            "EnableThumbnailDownsample": "true",
            "EnableHighResImagesForRenderingThumbnail": "true",
            "EnableUrlForSuggestionPayload": "true",
            "UseDFSOnePrompt": "false"
        }
    })

    headers = {
        'authority': 'designerapp.officeapps.live.com',
        'accept': '*/*',
        'accept-language': 'vi,en-US;q=0.9,en;q=0.8',
        'audiencegroup': 'Production',
        'caller': 'DesignerApp',
        'clientbuild': '1.0.20231005.5',
        'clientname': 'DesignerApp',
        'containerid': 'ed7f96e0-e423-42f0-8d7d-57823db6e6f2',
        'content-type': 'application/json',
        'filetoken': '420ff1d6-e1f4-4383-bf24-e5134f37dc43',
        'hostapp': 'DesignerApp',
        'origin': 'https://designer.microsoft.com',
        'platform': 'Web',
        'referer': 'https://designer.microsoft.com/',
        'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'sessionid': 'f78cc69a-65e6-46d5-a6f4-ef88483852c0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'userid': 'a6ece104d62ec61f',
        'usertype': 'MSA',
        'x-correlation': '7aab8ee3-f066-46ee-967e-978f61a9d301',
        'x-correlation-id': '7aab8ee3-f066-46ee-967e-978f61a9d301',
        'x-dc-hint': 'KoreaCentral',
        'x-req-start': '1029201.1999999881'
    }
    return url, headers, payload


def text_to_image(url, headers, payload, prompt):
    # Parse the JSON string into a Python dictionary
    payload = json.loads(payload)

    # Set the DesignQuery field to your desired text
    design_query_text = prompt
    payload["Hints"]["DesignQuery"] = design_query_text

    payload = json.dumps(payload)

    return requests.request("POST", url, headers=headers, data=payload)


def find_substring_between_bytes(data, start_string, end_string):
    start_bytes = start_string.encode('utf-8')
    end_bytes = end_string.encode('utf-8')
    start_index = data.find(start_bytes)
    end_index = data.find(end_bytes, start_index + len(start_bytes))

    if start_index != -1 and end_index != -1:
        return data[start_index + len(start_bytes):end_index]
    else:
        return None


def save_results(response):
    image_paths = []
    for i in range(response.content.count(b'Content-Type: image/jpeg')):
        end_str = "\r\n--...Boundary_inner" + str(i) + "--"
        start_str = "\r\n--...Boundary_inner" + str(i) + "\r\n" + "Content-Type: image/jpeg\r\n\r\n"
        # \r\n--...Boundary_inner1\r\nContent-Type: image/jpeg
        result = find_substring_between_bytes(response.content, start_str, end_str)
        if result is not None:
            image = Image.open(BytesIO(result))
            file_path = 'C:/Users/thanh/PycharmProjects/flaskProject/flaskblog/static/database/ImageGenerate' + str(
                i) + '.png'
            image.save(file_path)
            image_paths.append('ImageGenerate' + str(i) + '.png')
    return image_paths


if __name__ == '__main__':
    # input your prompt
    prompt = " A photorealistic image of a JBL Roofing and Construction employee repairing a roof after a storm"
    url, headers, payload = initial_configs()
    response = text_to_image(url, headers, payload, prompt)
    # save images that generated
    save_results(response)
