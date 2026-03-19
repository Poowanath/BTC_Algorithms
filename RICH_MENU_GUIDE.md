# คู่มือการสร้าง Rich Menu สำหรับ LINE Bot

## Rich Menu คืออะไร?
Rich Menu คือเมนูที่แสดงด้านล่างของหน้าแชท LINE Bot ซึ่งผู้ใช้สามารถกดปุ่มต่างๆ เพื่อส่งคำสั่งหรือเปิด URL ได้

## ขั้นตอนการสร้าง Rich Menu

### 1. สร้างรูปภาพ Rich Menu

```bash
python create_rich_menu_image.py
```

รูปภาพจะถูกสร้างที่ `picture/rich_menu.png` (ขนาด 2500x1686 pixels)

หรือคุณสามารถออกแบบรูปภาพเองด้วยโปรแกรมอย่าง:
- Photoshop
- Canva
- Figma

**ข้อกำหนดรูปภาพ:**
- ขนาด: 2500x1686 pixels (แบบเต็ม) หรือ 2500x843 pixels (แบบครึ่ง)
- ไฟล์: PNG หรือ JPEG
- ขนาดไฟล์: ไม่เกิน 1 MB

### 2. จัดการ Rich Menu

```bash
python bot/rich_menu.py
```

เมนูที่มี:
1. **สร้าง Rich Menu ใหม่** - สร้าง Rich Menu และอัปโหลดรูปภาพ
2. **แสดงรายการ Rich Menu** - ดู Rich Menu ที่มีอยู่
3. **ลบ Rich Menu** - ลบ Rich Menu ที่ไม่ต้องการ

### 3. ตั้งค่า Environment Variables

ตรวจสอบว่าไฟล์ `.env` มี:

```env
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token
LINE_CHANNEL_SECRET=your_channel_secret
```

## โครงสร้าง Rich Menu ของเรา

### แถวบน (3 ปุ่ม)
```
┌─────────────┬─────────────┬─────────────┐
│   ราคา BTC  │    กราฟ     │   พยากรณ์   │
│     💰      │     📊      │     🔮      │
└─────────────┴─────────────┴─────────────┘
```

### แถวล่าง (4 ปุ่ม)
```
┌────────┬────────┬────────┬────────────┐
│ Trend  │  Mean  │  Grid  │ เปรียบเทียบ │
│   📈   │   🔄   │   ⚡   │     🏆     │
└────────┴────────┴────────┴────────────┘
```

## พื้นที่ปุ่ม (Bounds)

Rich Menu ขนาด 2500x1686 แบ่งเป็น:

### แถวบน (y: 0-843)
- **ราคา BTC**: x=0, y=0, width=833, height=843
- **กราฟ**: x=833, y=0, width=834, height=843
- **พยากรณ์**: x=1667, y=0, width=833, height=843

### แถวล่าง (y: 843-1686)
- **Trend**: x=0, y=843, width=625, height=843
- **Mean Reversion**: x=625, y=843, width=625, height=843
- **Grid Trading**: x=1250, y=843, width=625, height=843
- **เปรียบเทียบ**: x=1875, y=843, width=625, height=843

## การทดสอบ

1. เปิด LINE แอพ
2. เพิ่มเพื่อน Bot ของคุณ
3. Rich Menu จะแสดงด้านล่างหน้าแชท
4. ลองกดปุ่มต่างๆ เพื่อทดสอบ

## Tips

- ใช้สีที่สะดุดตาแต่ไม่จ้าเกินไป
- ใส่ไอคอนหรือรูปภาพเพื่อให้เข้าใจง่าย
- ข้อความควรสั้นและชัดเจน
- ทดสอบบนมือถือจริงเพื่อดูขนาดที่เหมาะสม

## ตัวอย่างการใช้งาน API

### สร้าง Rich Menu ด้วยโค้ด

```python
from bot.rich_menu import create_rich_menu, upload_rich_menu_image, set_default_rich_menu

# สร้าง Rich Menu
rich_menu_id = create_rich_menu(channel_access_token)

# อัปโหลดรูปภาพ
upload_rich_menu_image(channel_access_token, rich_menu_id, "picture/rich_menu.png")

# ตั้งเป็นค่าเริ่มต้น
set_default_rich_menu(channel_access_token, rich_menu_id)
```

## เอกสารอ้างอิง

- [LINE Messaging API - Rich Menu](https://developers.line.biz/en/docs/messaging-api/using-rich-menus/)
- [Rich Menu Design Guide](https://developers.line.biz/en/docs/messaging-api/rich-menu-design-guide/)
