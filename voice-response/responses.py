import requests
import ast
import wget
import os
from playsound import playsound
import time

responses_text = \
{
    '0_angry': [ 
        'Ấy ơi bớt giận. Khơm đao Khơm đao.', \
        'Ai đã làm ấy giận đấy?', \
        'Chuyện gì làm boss của tôi giận thế!', \
        'Người ơi, đừng giận nữa.', \
        'Người ơi bớt giận. Khơm đao đi nào.' \
    ],
    
    '1_disgust': [
        'Địt gớt. Địt gớt. Địt gớt. Địt gớt. Địt gớt', \
        'Địt gớt 2. Địt gớt 2. Địt gớt 2. Địt gớt 2. Địt gớt 2', \
        'Khi điều khoản luận tội được quá bán nghị sĩ Hạ viện thông qua.' \
    ],
    
    '2_fear': [ \
        'Trông anh có vẻ sợ. Có chuyện gì thế anh yêu?', \
        'Anh yêu đừng sợ, có em này', \
        'một số nghị sĩ đảng Cộng hòa cũng tuyên bố sẽ ủng hộ.' \
    ],
    
    '3_happy': [
        'Có chuyện gì mà vui thế anh?', \
        'Trông anh hôm nay vui lắm, có thể chia sẻ với em không?', \
        'Hiện chưa rõ thời gian chính xác Hạ viện tiến hành cuộc bỏ phiếu.', \
        'Ít nhất 210 hạ nghị sĩ Dân chủ nhất trí luận tội Trump.' \
    ],
    
    '4_sad': [
        'Ai đã khiến anh buồn? Hãy nói cho em biết!', \
        'Trông anh buồn làm em cũng buồn theo. Có chuyện gì thế anh?', \
        'Trông anh buồn quá, có phải lại nhớ người yêu cũ không?' \
    ],

    '5_surprise': [
        'Cuộc bỏ phiếu được tiến hành với điều khoản.', \
        'xem xét bãi nhiệm Tổng thống Donald Trump do hơn 210 hạ nghị.', \
        'sĩ Dân chủ đồng bảo trợ, trong đó cáo buộc Trump' \
    ],
    
    '6_neutral': [
        'đã kích động một cuộc tấn công vào chính phủ bằng cách xúi giục đám đông.', \
        'ủng hộ ông cố ngăn quốc hội chứng nhận.', \
        'chiến thắng của Tổng thống đắc của Joe Biden.' \
    ]
}


url = 'https://api.fpt.ai/hmi/tts/v5'

headers = {
    'api-key': 'cOtYkSMBr4XyjE6M13ckHgQNSS55q84M',
    'speed': '',
    'voice': 'banmai'
}

print(type(responses_text.items()))

for label, payloads in responses_text.items():
    print(f'------------------------------\nDEBUG: label = {label}\n')
    count = 0
    print(f'DEBUG: payloads = {payloads}')
    for payload in payloads:
        print(f'DEBUG: payload={payload}')
        response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)
        print(f'DEBUG: response = {response}')
        response_dict = ast.literal_eval(response.text)
        print(f'response_dict: {response_dict}')

        path_to_save = f'{label}_response_{count}.mp3'
        filename = wget.download(response_dict['async'], out=path_to_save)
        
        count += 1

        playsound(filename)



'''
payload = 'Ấy ơi bớt giận. Khơm đao Khơm đao.'
headers = {
    'api-key': 'cOtYkSMBr4XyjE6M13ckHgQNSS55q84M',
    'speed': '',
    'voice': 'banmai'
}

response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

response_dict = ast.literal_eval(response.text)

print(response_dict)

doc = requests.get(response_dict['async'])

with open('myfile.mp3', 'wb') as f:
    f.write(doc.content)
    
'''