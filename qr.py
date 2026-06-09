import qrcode

# QR 코드에 담을 링크나 텍스트
data = "https://lens-mosaic-1045259343465.asia-northeast1.run.app"

# QR 코드 생성
img = qrcode.make(data)

# 이미지 파일로 저장
img.save("my_qrcode.png")
print("QR 코드가 성공적으로 생성되었습니다!")