import os

def create_cds_config():
    # Tu dong xac dinh thu muc nguoi dung (C:\Users\Pham Ngoc Minh)
    home_dir = os.path.expanduser("~")
    config_path = os.path.join(home_dir, ".cdsapirc")

    # Thong tin cau hinh (Thay the chuoi key bang Token thuc te cua ban)
    url_line = "url: https://cds.climate.copernicus.eu/api"
    key_line = "key: 643c5e92-3e00-48c2-aeb9-70e8f01d5a12" 

    config_content = f"{url_line}\n{key_line}\n"

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Tao file cau hinh thanh cong tai: {config_path}")
    except Exception as e:
        print(f"Co loi xay ra khi tao file: {e}")

if __name__ == "__main__":
    create_cds_config()
