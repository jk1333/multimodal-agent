import json
import sys
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def download_and_modify_agent_card(target_url):
    # URL 끝의 슬래시(/) 처리 후 최종 요청 URL 생성
    base_url = target_url if target_url.endswith('/') else f"{target_url}/"
    request_url = f"{base_url}.well-known/agent-card.json"
    
    print(f"요청 중: {request_url}")
    
    try:
        # 브라우저처럼 보이도록 User-Agent 추가 (일부 서버의 차단 방지)
        req = Request(request_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        # 1. URL 요청 및 데이터 다운로드
        with urlopen(req) as response:
            raw_data = response.read().decode('utf-8')
            json_data = json.loads(raw_data)
        
        # 2. root의 'url' 키값을 입력받은 target_url로 교체
        json_data['url'] = target_url
        
        # 3. 'agent-card.json' 파일로 저장 (들여쓰기 2칸으로 깔끔하게 포맷팅)
        output_filename = "agent-card.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"성공: '{output_filename}' 파일이 생성되었고 url 값이 '{target_url}'로 교체되었습니다.")
        
    except HTTPError as e:
        print(f"HTTP 오류 발생: {e.code} - {e.reason}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"URL 오류 발생: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("오류: 다운로드한 데이터가 올바른 JSON 형식이 아닙니다.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # 인자값(URL) 확인
    if len(sys.argv) < 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"사용법: python {script_name} <URL>")
        print(f"예시: python {script_name} https://example.com")
        sys.exit(1)
        
    user_url = sys.argv[1]
    download_and_modify_agent_card(user_url)