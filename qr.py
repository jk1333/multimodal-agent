import argparse
import qrcode


def main():
    # 커맨드 라인 인자 설정을 위한 파서 생성
    parser = argparse.ArgumentParser(
        description="터미널에서 QR 코드를 생성하는 프로그램입니다."
    )

    # 필수 인자: QR 코드에 담을 데이터 (링크 또는 텍스트)
    parser.add_argument(
        "data", type=str, help="QR 코드에 담을 URL 또는 텍스트를 입력하세요."
    )

    # 선택 인자: 저장할 파일 이름 (기본값: qrcode.png)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="qrcode.png",
        help="저장할 이미지 파일명을 입력하세요. (기본값: qrcode.png)",
    )

    # 인자 파싱
    args = parser.parse_args()

    try:
        # QR 코드 생성
        img = qrcode.make(args.data)
        # 이미지 파일로 저장
        img.save(args.output)
        print(f" 성공: QR 코드가 '{args.output}' 파일로 저장되었습니다!")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()