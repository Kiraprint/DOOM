// Custom VKR title page template for modern-g7-32
// Matches original Word/PDF title page layout

#let arguments(..args, year: auto, city: "Санкт-Петербург") = {
  let args = args.named()
  
  args.organization = args.at("organization", default: (full: none, short: none))
  args.department = args.at("department", default: none)
  args.faculty = args.at("faculty", default: none)
  args.program = args.at("program", default: none)
  args.direction = args.at("direction", default: none)
  args.manager = args.at("manager", default: none)
  args.performer = args.at("performer", default: none)
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
  city: "Санкт-Петербург",
  year: auto,
  ..rest
) = {
  let resolved-year = if year == auto { 2026 } else { year }

  set text(font: "Times New Roman", size: 14pt, lang: "ru", hyphenate: false)

  // === БЛОК 1: Министерство (12pt bold, одна строка) ===
  align(center, text(weight: "bold", "Министерство науки и высшего образования Российской Федерации"))
  
  // === БЛОК 2: Организация (14pt CAPS bold) + короткое название ===
  if organization.full != none {
    let org-short = organization.at("short", default: none)
  
    align(center, stack(dir: ttb, spacing: 1em,
      text(size: 14pt, weight: "bold", "ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ"),
      text(size: 14pt, weight: "bold", "ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ"),
      if org-short != none {
        text(size: 12pt, weight: "bold", org-short)
      }
    ))
    v(60pt)
  }

  // === БЛОК 3: Факультет/Программа/Направление (13pt, слева) ===
  if faculty != none {
    align(left, stack(dir: ltr,
      text(size: 13pt, weight: "bold", "Факультет: "),
      text(size: 13pt, weight: "regular", faculty)))
  }

  if program != none {
    align(left, stack(dir: ttb, spacing: 1em,
      text(size: 13pt, weight: "bold", "Образовательная программа:"),
      text(size: 13pt, weight: "regular", program)))
  }

  if direction != none {
    align(left, stack(dir: ttb, spacing: 1em,
      stack(dir: ltr,
        text(size: 13pt, weight: "bold", "Направление подготовки (специальность): "),
        text(size: 13pt, weight: "regular", direction.number),
      ),
      text(size: 13pt, weight: "regular", direction.name),
    ))
    v(30pt)
  }

  // === БЛОК 4: ОТЧЁТ (18pt, разрядка букв) ===
  align(center, text(18pt, "О") + text(0.4em, " ") + text(18pt, "Т") + text(0.4em, " ")
    + text(18pt, "Ч") + text(0.4em, " ") + text(18pt, "Ё") + text(0.4em, " ") + text(18pt, "Т"))
  v(6pt)

  // === БЛОК 5: "по выпускной квалификационной работе" + Тема ===
  align(center, text(size: 14pt, "по выпускной квалификационной работе"))
  v(6pt)
  align(center, text(size: 14pt,
    "Тема: Разработка архитектуры нейросетевого агента на основе слоёв\n"
    + "Mamba-2 для решения задач в среде VizDoom"))
  v(150pt)

  // === БЛОК 6: Исполнитель и руководитель ===
  if performer != none {
    align(right, text(size: 14pt, "Выполнил: " + performer.name))
  }
  if manager != none {
    align(right, text(size: 14pt, "Научный руководитель: " + manager.name))
  }

  // pagebreak()
}
