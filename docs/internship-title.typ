// internship-title.typ — Титульная страница отчёта о прохождении стажировки
// Адаптировано из practice-title.typ для отчёта о стажировке в ИЦ СИИП

#let arguments(..args, year: auto, city: "Санкт-Петербург") = {
  let args = args.named()
  args.organization = args.at("organization", default: (full: none, short: none))
  args.faculty = args.at("faculty", default: none)
  args.program = args.at("program", default: none)
  args.direction = args.at("direction", default: none)
  args.manager = args.at("manager", default: none)
  args.performer = args.at("performer", default: none)
  args.report_type = args.at("report_type", default: none)
  args.city = args.at("city", default: city)
  if year == auto {
    args.date = datetime.today()
  }
  return args
}

#let template(
  organization: (full: none, short: none),
  faculty: none,
  program: none,
  direction: (number: none, name: none),
  manager: (position: none, name: none),
  performer: (name: none, position: none),
  report_type: "стажировки",
  city: "Санкт-Петербург",
  year: auto,
  ..rest
) = {
  let resolved-year = if year == auto { 2026 } else { year }

  set text(font: "Times New Roman", size: 14pt, lang: "ru", hyphenate: false)

  // === БЛОК 1: Министерство ===
  align(center, text(weight: "bold", "Министерство науки и высшего образования Российской Федерации"))

  // === БЛОК 2: Организация ===
  if organization.full != none {
    let org-short = organization.at("short", default: none)
    align(center, stack(dir: ttb, spacing: 1em,
      text(size: 14pt, weight: "bold", "ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ"),
      text(size: 14pt, weight: "bold", "ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ"),
      if org-short != none {
        text(size: 12pt, weight: "bold", org-short)
      }
    ))
    v(30pt)
  }

  // === БЛОК 3: Структурное подразделение (Исследовательский центр) ===
  align(center, text(size: 13pt, weight: "bold",
    "Исследовательский центр «Сильный искусственный интеллект в промышленности»"))
  align(center, text(size: 12pt, "Институт искусственного интеллекта"))

  v(20pt)


  // === БЛОК 5: ОТЧЁТ (18pt, разрядка) ===
  align(center, text(18pt, "О") + text(0.4em, " ")
    + text(18pt, "Т") + text(0.4em, " ") + text(18pt, "Ч") + text(0.4em, " ")
    + text(18pt, "Ё") + text(0.4em, " ") + text(18pt, "Т"))
  v(6pt)

  // === БЛОК 6: Вид отчёта ===
  align(center, text(size: 14pt, "о прохождении " + report_type))
  v(6pt)
  align(center, text(size: 14pt,
    "Тема: Разработка многомодульной RAG-системы с иерархическим\n"
    + "выбором инструментов для управления городской инфраструктурой\n"
    + "на основе LLM-агентов"))
  v(150pt)

  // === БЛОК 7: Исполнитель и руководитель ===
  if performer != none {
    align(right, text(size: 14pt, "Выполнил: " + performer.name))
  }
  if manager != none {
    align(right, text(size: 14pt, "Научный руководитель: " + manager.name))
  }

}
