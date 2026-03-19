"""
LINE Rich Menu Management
สร้างและจัดการ Rich Menu สำหรับ LINE Bot
"""

import os
import json
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    RichMenuRequest,
    RichMenuSize,
    RichMenuArea,
    RichMenuBounds,
    MessageAction,
    URIAction,
)


def create_rich_menu(channel_access_token: str) -> str:
    """
    สร้าง Rich Menu สำหรับ BTC Trading Bot
    
    Returns:
        rich_menu_id: ID ของ Rich Menu ที่สร้างขึ้น
    """
    configuration = Configuration(access_token=channel_access_token)
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        # สร้าง Rich Menu (ขนาด 2500x1686 pixels - แบบเต็ม)
        rich_menu = RichMenuRequest(
            size=RichMenuSize(width=2500, height=1686),
            selected=True,
            name="BTC Trading Bot Menu",
            chat_bar_text="เมนู",
            areas=[
                # แถวบน - ซ้าย: ราคา BTC
                RichMenuArea(
                    bounds=RichMenuBounds(x=0, y=0, width=833, height=843),
                    action=MessageAction(text="ราคา")
                ),
                # แถวบน - กลาง: กราฟ
                RichMenuArea(
                    bounds=RichMenuBounds(x=833, y=0, width=834, height=843),
                    action=MessageAction(text="กราฟ")
                ),
                # แถวบน - ขวา: พยากรณ์
                RichMenuArea(
                    bounds=RichMenuBounds(x=1667, y=0, width=833, height=843),
                    action=MessageAction(text="predict")
                ),
                # แถวล่าง - ซ้าย: Trend
                RichMenuArea(
                    bounds=RichMenuBounds(x=0, y=843, width=625, height=843),
                    action=MessageAction(text="trend")
                ),
                # แถวล่าง - กลางซ้าย: Mean Reversion
                RichMenuArea(
                    bounds=RichMenuBounds(x=625, y=843, width=625, height=843),
                    action=MessageAction(text="mean")
                ),
                # แถวล่าง - กลางขวา: Grid Trading
                RichMenuArea(
                    bounds=RichMenuBounds(x=1250, y=843, width=625, height=843),
                    action=MessageAction(text="grid")
                ),
                # แถวล่าง - ขวา: เปรียบเทียบ
                RichMenuArea(
                    bounds=RichMenuBounds(x=1875, y=843, width=625, height=843),
                    action=MessageAction(text="compare")
                ),
            ]
        )
        
        # สร้าง Rich Menu
        response = line_bot_api.create_rich_menu(rich_menu_request=rich_menu)
        rich_menu_id = response.rich_menu_id
        
        print(f"✅ สร้าง Rich Menu สำเร็จ: {rich_menu_id}")
        return rich_menu_id


def upload_rich_menu_image(channel_access_token: str, rich_menu_id: str, image_path: str):
    """
    อัปโหลดรูปภาพสำหรับ Rich Menu
    
    Args:
        channel_access_token: LINE Channel Access Token
        rich_menu_id: Rich Menu ID
        image_path: path ของรูปภาพ (ต้องเป็น PNG หรือ JPEG, ขนาด 2500x1686 pixels)
    """
    import requests
    
    url = f"https://api-data.line.me/v2/bot/richmenu/{rich_menu_id}/content"
    headers = {
        "Authorization": f"Bearer {channel_access_token}",
        "Content-Type": "image/png"
    }
    
    with open(image_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f)
    
    if response.status_code == 200:
        print(f"✅ อัปโหลดรูปภาพสำเร็จ")
    else:
        print(f"❌ อัปโหลดรูปภาพล้มเหลว: {response.text}")


def set_default_rich_menu(channel_access_token: str, rich_menu_id: str):
    """
    ตั้งค่า Rich Menu เป็นค่าเริ่มต้นสำหรับผู้ใช้ทุกคน
    """
    configuration = Configuration(access_token=channel_access_token)
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.set_default_rich_menu(rich_menu_id=rich_menu_id)
        print(f"✅ ตั้งค่า Rich Menu เป็นค่าเริ่มต้นสำเร็จ")


def delete_rich_menu(channel_access_token: str, rich_menu_id: str):
    """ลบ Rich Menu"""
    configuration = Configuration(access_token=channel_access_token)
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.delete_rich_menu(rich_menu_id=rich_menu_id)
        print(f"✅ ลบ Rich Menu สำเร็จ: {rich_menu_id}")


def list_rich_menus(channel_access_token: str):
    """แสดงรายการ Rich Menu ทั้งหมด"""
    configuration = Configuration(access_token=channel_access_token)
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        response = line_bot_api.get_rich_menu_list()
        
        print(f"\n📋 Rich Menu ทั้งหมด ({len(response.richmenus)} รายการ):")
        for menu in response.richmenus:
            print(f"  - ID: {menu.rich_menu_id}")
            print(f"    Name: {menu.name}")
            print(f"    Chat Bar Text: {menu.chat_bar_text}")
            print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    
    if not channel_access_token:
        print("❌ กรุณาตั้งค่า LINE_CHANNEL_ACCESS_TOKEN ใน .env")
        exit(1)
    
    print("🤖 LINE Rich Menu Management")
    print("=" * 50)
    print("1. สร้าง Rich Menu ใหม่")
    print("2. แสดงรายการ Rich Menu")
    print("3. ลบ Rich Menu")
    print("=" * 50)
    
    choice = input("เลือกคำสั่ง (1-3): ").strip()
    
    if choice == "1":
        # สร้าง Rich Menu
        rich_menu_id = create_rich_menu(channel_access_token)
        
        # ถามว่าจะอัปโหลดรูปภาพหรือไม่
        upload = input("\nต้องการอัปโหลดรูปภาพหรือไม่? (y/n): ").strip().lower()
        if upload == "y":
            image_path = input("ใส่ path ของรูปภาพ (เช่น picture/rich_menu.png): ").strip()
            if os.path.exists(image_path):
                upload_rich_menu_image(channel_access_token, rich_menu_id, image_path)
                
                # ตั้งเป็นค่าเริ่มต้น
                set_default = input("\nตั้งเป็น Rich Menu เริ่มต้นหรือไม่? (y/n): ").strip().lower()
                if set_default == "y":
                    set_default_rich_menu(channel_access_token, rich_menu_id)
            else:
                print(f"❌ ไม่พบไฟล์: {image_path}")
    
    elif choice == "2":
        list_rich_menus(channel_access_token)
    
    elif choice == "3":
        list_rich_menus(channel_access_token)
        rich_menu_id = input("\nใส่ Rich Menu ID ที่ต้องการลบ: ").strip()
        confirm = input(f"ยืนยันการลบ {rich_menu_id}? (y/n): ").strip().lower()
        if confirm == "y":
            delete_rich_menu(channel_access_token, rich_menu_id)
    
    else:
        print("❌ คำสั่งไม่ถูกต้อง")
