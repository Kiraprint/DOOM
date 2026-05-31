#!/usr/bin/env bash
set -euo pipefail

# ─── Yandex Neurodetector CLI — проверка PDF на ИИ-шность ───
# Usage: ./check-ai.sh [file.pdf]
# Default: main.pdf

FILE="${1:-main.pdf}"
COOKIE_FILE="/tmp/yandex-neuro-cookies.txt"

# ─── Cookies (из браузера, живут ~месяц) ───
COOKIES='is_gdpr=0; is_gdpr_b=CIrYeRD/+wIoAg==; yandexuid=2011386511774699380; yashr=7217822711774699380; L=ZQpIaXtpX0MEZwYHDH9BQFJOUAcNBENdXzo1Kl0FSxQ9RBQ=.1774852085.1801173.389195.b87d3f3a47e7c6a44f5cf9db905f10e4; yandex_login=kirill.fdrn; yuidss=2011386511774699380; Session_id=3:1777631191.5.0.1774852085763:bSOKLg:5ddc.1.2:1|295017209.-1.2.0:3.3:1774852085.6:2088908079.7:1774852085|3:11875238.870513.TO0aL-x0q4rVubZhpxQoa3JmHVU; sessar=1.1781281.CiBy5nLareogVRgQdyd38ev1YyYa2OUUumlIDZycRNrXQg.pqLxBp3E6BQMKDwVCX3p7XCoNkaQLprDGaSAXwQrFEs; sessionid2=3:1777631191.5.0.1774852085763:bSOKLg:5ddc.1.2:1|295017209.-1.2.0:3.3:1774852085.6:2088908079.7:1774852085|3:11875238.870513.fakesign0000000000000000000; yp=2090212085.udn.cDpLaXJhcHJpbnQ%3D#1778344580.szm.1_25:2560x1440:2048x1239; _yasc=7fFVyPAiSbB8o6QhU9vMlsiPZ4ljbgetlTFy7+gS5c1EmXzkb8hMHLGCIwu8qmZUfAyKknhVaLg=.MTc3OTE2OTI4NTIyNA==; pi=TAQzL1uPV6C47QwLKNUWnmA90+OHS05q5JRlTrhM1zrlx6IgyPD3NSKyuPGh3px1fFuII5kjrzFh6GWCZXvrGrjUT9M=; i=zEEC+NfEzocCPQX3ULkVYEuVuLSWRtZN8sZkFkuiUxNgViunrKht1TLNUM0ctVUoIgNaPcCYTjwJ6VjjbjMzkd3UQ1g=; bh=EjkiQ2hyb21pdW0iO3Y9IjE0OCIsICJCcmF2ZSI7dj0iMTQ4IiwgIk5vdC9BKUJyYW5kIjt2PSI5OSIaBSJ4ODYiKgI/MDICIiI6ByJMaW51eCJCAiIiSgQiNjQiUksiQ2hyb21pdW0iO3Y9IjE0OC4wLjAuMCIsICJCcmF2ZSI7dj0iMTQ4LjAuMC4wIiwgIk5vdC9BKUJyYW5kIjt2PSI5OS4wLjAuMCJaAj8wYKuY49AGahncyumIDvKst6UL+/rw5w3r//32D9OgzocI'

# ─── Проверка файла ───
if [ ! -f "$FILE" ]; then
    echo "❌ Файл не найден: $FILE"
    exit 1
fi

echo "🔍 Отправка $FILE на Yandex Neurodetector..."

# ─── curl с -F (сам формирует multipart) ───
# Реально шлём файл через -F, а не пустой --data-raw
RESPONSE=$(curl -s -w '\n%{http_code}' \
    'https://yandex.ru/lab/neurodetector/api/analyze/file' \
    -H 'accept: */*' \
    -H 'accept-language: en-US,en;q=0.8' \
    -H 'origin: https://yandex.ru' \
    -H 'referer: https://yandex.ru/lab/neurodetector' \
    -H 'sec-ch-ua: "Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"' \
    -H 'sec-ch-ua-arch: "x86"' \
    -H 'sec-ch-ua-bitness: "64"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-model: ""' \
    -H 'sec-ch-ua-platform: "Linux"' \
    -H 'sec-ch-ua-platform-version: ""' \
    -H 'sec-ch-ua-wow64: ?0' \
    -H 'sec-fetch-dest: empty' \
    -H 'sec-fetch-mode: cors' \
    -H 'sec-fetch-site: same-origin' \
    -H 'sec-gpc: 1' \
    -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36' \
    -b "$COOKIES" \
    -F "file=@$FILE;type=application/pdf")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ HTTP $HTTP_CODE"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi

# ─── Парсим результат ───
echo "$BODY" | python3 -c "
import json, sys

data = json.load(sys.stdin)
if 'results' in data:
    data = data['results']

stats = data.get('stats', {})
segments = data.get('segments', [])

total = stats.get('segments_count', len(segments))
ai = stats.get('AI_count', 0)
likely_ai = stats.get('LIKELY_AI_count', 0)
human = stats.get('HUMAN_count', 0)
likely_human = stats.get('LIKELY_HUMAN_count', 0)
pct = (ai + likely_ai) / total * 100 if total > 0 else 0

print()
print('╔══════════════════════════════════════╗')
print('║   Yandex Neurodetector — Результат  ║')
print('╠══════════════════════════════════════╣')
print(f'║  Всего сегментов:    {total:>3}            ║')
print(f'║  HUMAN:              {human:>3}            ║')
print(f'║  LIKELY_HUMAN:       {likely_human:>3}            ║')
print(f'║  LIKELY_AI:          {likely_ai:>3}            ║')
print(f'║  AI:                 {ai:>3}            ║')
print(f'║  AI+LIKELY_AI:       {ai+likely_ai:>3}/{total} = {pct:5.1f}%      ║')
print('╚══════════════════════════════════════╝')

if pct < 5:
    print('✅ Норма: менее 5%')
elif pct < 15:
    print('⚠️ Предупреждение: 5-15%')
else:
    print('❌ Критично: более 15%')

print()
print('Подозрительные сегменты:')
for i, s in enumerate(segments):
    label = s.get('label', '?')
    if label in ('AI', 'LIKELY_AI', 'LIKELY_HUMAN'):
        text = s.get('text', '')[:150]
        print(f'  [{i:>2}] {label:14s}  {text}')
"
