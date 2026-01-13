import requests
import base64
import os

API_URL = "http://127.0.0.1:8000/restore" 
INPUT_IMAGE = "test.jpg"           
OUTPUT_IMAGE = "sonuc1.png"

def test_api():
    if not os.path.exists(INPUT_IMAGE):
        print(f"'{INPUT_IMAGE}' dosyası bulunamadı!")
        return

    print(f"1. '{INPUT_IMAGE}' okunuyor...")
    
    with open(INPUT_IMAGE, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {"image_base64": base64_string}

    print("2. Modele gönderildi.")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # Anahtar ismi değişti: 'restored_base64'
            if "restored_base64" in data:
                restored_b64 = data["restored_base64"]
                
                with open(OUTPUT_IMAGE, "wb") as fh:
                    fh.write(base64.b64decode(restored_b64))
                    
                print(f"3. Başarılı: '{OUTPUT_IMAGE}'")
            else:
                print("Hata: Beklenen veri gelmedi.", data)
            
        else:
            print(f"HATA: {response.status_code} döndü.")
            print("Detay:", response.text)

    except requests.exceptions.ConnectionError:
        print("HATA: Sunucuya bağlanılamadı.")

if __name__ == "__main__":
    test_api()