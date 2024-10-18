import json
import difflib
import sys

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def compare_json(file1, file2):
    # JSON 파일 로드
    json1 = load_json(file1)
    json2 = load_json(file2)

    # JSON을 문자열로 변환 (정렬하여 키 순서 차이 무시)
    str1 = json.dumps(json1, sort_keys=True, indent=2).splitlines()
    str2 = json.dumps(json2, sort_keys=True, indent=2).splitlines()

    # 차이점 계산
    differ = difflib.Differ()
    diff = list(differ.compare(str1, str2))

    # 차이점 출력
    for line in diff:
        if line.startswith('- '):
            print('\033[91m' + line + '\033[0m')  # 빨간색으로 출력
        elif line.startswith('+ '):
            print('\033[92m' + line + '\033[0m')  # 초록색으로 출력
        elif line.startswith('? '):
            continue  # '?' 로 시작하는 라인은 무시
        else:
            print(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1.json> <file2.json>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_json(file1, file2)