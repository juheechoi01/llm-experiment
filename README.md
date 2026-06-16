# Folder for 학위논문!
## llm-대화 실험 prompt & application

## 대화 완료 후 리다이렉트

조건 URL(`/condition1` 등)에서 참가자가 3회 이상 대화한 뒤 `대화 완료` 버튼을 누르면
`REDIRECT_BASE_URL` 환경변수에 지정한 주소로 이동합니다.

이동 URL에는 자동으로 아래 파라미터가 붙습니다.

- `panel_id`: 접속 URL의 `panel_id` 값
- `status`: `001`

예시:

```text
REDIRECT_BASE_URL=https://example.com/survey/complete
```

참가자가 아래처럼 접속했다면:

```text
https://your-app.up.railway.app/condition1?panel_id=abc123
```

완료 후 아래 주소로 이동합니다.

```text
https://example.com/survey/complete?panel_id=abc123&status=001
```
