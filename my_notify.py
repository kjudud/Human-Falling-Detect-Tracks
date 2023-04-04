import requests

def send_message():
    try:

        TARGET_URL = 'https://notify-api.line.me/api/notify'
        TOKEN = 'vDMl75kkrSo9jV4iFVYd4fMM2FaDtHxLotEE7RIxLvG'

        response = requests.post(
            TARGET_URL,
            headers={
                'Authorization': 'Bearer ' + TOKEN
            },
            data={
                'message': 'extract complete'
            }
        )

    except Exception as ex:
        print(ex)