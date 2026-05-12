// main.typ — ВКР: Разработка архитектуры нейросетевого агента
// на основе слоев Mamba-2 для решения задач в среде VizDoom

#import "@preview/modern-g7-32:0.2.0": gost, abstract, appendixes
#import "vkr-title.typ"

#show: gost.with(
  title-template: vkr-title.template,
  title-arguments: vkr-title.arguments,
  organization: (
    full: "Федеральное государственное автономное образовательное учреждение высшего образования",
    short: "«Национальный исследовательский университет ИТМО» (Университет ИТМО)"
  ),
  faculty: "Факультета прикладной информатики",
  program: "Программирование в инфокоммуникационных системах",
  direction: (
    number: "09.03.03",
    name: "Прикладная информатика",
  ),
  manager: (
    name: "Гусарова Н. Ф.",
  ),
  performer: (
    name: "Федорин К.В.",
  ),
  city: "Санкт-Петербург",
  year: auto,
  margin: (left: 30mm, right: 15mm, rest: 20mm),
)

// Переопределяем стили ПОСЛЕ gost.with() чтобы шаблон не перезаписал

// Межстрочный интервал 1.5, без отступов между абзацами
#set par(leading: 1em, spacing: 1em)

// Убираем отступы вокруг заголовков (по умолчанию 2em сверху и снизу)
#show heading: set block(below: 1em, above: 1em)

// Нумерация: N / N.N / N.N.N / N.N.N.N
// Структурные заголовки (Введение, Заключение) — без номера (обрабатываются в gost)
#let heading-numbering(..nums) = {
  let n = nums.pos()
  if n.len() == 1 { numbering("1", n.at(0)) }
  else { numbering("1.1", ..n) }
}
#set heading(numbering: heading-numbering)
// Заголовки 3+ уровня — обычным шрифтом (не жирным)
#show heading.where(level: 3): set text(weight: "regular")
#show heading.where(level: 4): set text(weight: "regular")

#abstract(
  "нейросетевой агент",
  "визуальная навигация",
  "обучение с подкреплением",
  "Mamba-2",
  "ViZDoom",
  "рейнфорсмент-обучение",
  "селективные модели пространства состояний",
)[
  Сравниваются архитектуры памяти для RL-агента в ViZDoom: GRU, Transformer, Mamba-2 и Perceiver IO. Программный комплекс на Sample Factory позволяет менять модуль памяти без затрагивания энкодера и декодера; для Mamba-2 дополнительно реализована сериализация двух внутренних состояний (conv_state + ssm_state) в rnn_states и gradient checkpointing. Поиск гиперпараметров Mamba-2 выполнен байесовской оптимизацией (85 trials). Эксперименты на 50 млн шагов показали, что GRU с гиперпараметрами, найденными для Mamba-2, достигает 17.86 ± 2.11 (seed3: 20.15) против 13.58 ± 1.02 у стокового GRU; сама Mamba-2 даёт 16.20 ± 2.01. GRU 250M упирается в плато около 22.45 после 150M шагов; Mamba-1, Transformer и Perceiver IO существенно уступают.
]

#outline(title: "Содержание")

= Термины и определения

#set list(marker: "—")
- _Агент_ — программа, взаимодействующая со средой, выбирающая действия на основе наблюдений и получающая награду.
- _Среда (окружение)_ — внешняя система, с которой взаимодействует агент; в данной работе — ViZDoom.
- _Наблюдение (observation)_ — входные данные агента на каждом шаге; в визуальном RL — кадр изображения.
- _Награда (reward)_ — численный сигнал от среды, который агент стремится максимизировать.
- _Эпизод_ — последовательность шагов агента от начала до терминального состояния (смерть, завершение уровня).
- _POMDP_ — марковский процесс принятия решений с частичной наблюдаемостью.
- _Архитектура памяти_ — модуль нейросети, обрабатывающий последовательности наблюдений и поддерживающий скрытое состояние.
- _Гиперпараметры (HP)_ — параметры модели и алгоритма обучения, задаваемые до начала эксперимента.
- _Байесовская оптимизация (HPO)_ — метод поиска гиперпараметров, строящий вероятностную модель целевой функции.
- _Gradient checkpointing_ — техника уменьшения потребления видеопамяти за счёт пересчёта промежуточных активаций на обратном проходе.
- _Селективные SSM_ — модели пространства состояний, где параметры динамики зависят от входного сигнала.
#set list(marker: "-")

= Перечень сокращений и обозначений

#set list(marker: "—")
- _APPO_ — Asynchronous Proximal Policy Optimization
- _BPTT_ — Backpropagation Through Time
- _FPS_ — Frames Per Second
- _GPU_ — Graphics Processing Unit
- _GRU_ — Gated Recurrent Unit
- _HPO_ — Hyperparameter Optimization
- _LSTM_ — Long Short-Term Memory
- _MDP_ — Markov Decision Process
- _POMDP_ — Partially Observable Markov Decision Process
- _PPO_ — Proximal Policy Optimization
- _RL_ — Reinforcement Learning
- _SAC_ — Soft Actor-Critic
- _SSM_ — State Space Model
- _TPE_ — Tree-structured Parzen Estimator
- _VRAM_ — Video Random Access Memory
#set list(marker: "-")

= Введение

Агент действует в трёхмерном мире. Он видит только пиксели с камеры — ни координат, ни карты, ни полного состояния среды у него нет. Такая постановка называется частично наблюдаемым марковским процессом принятия решений (POMDP), и она стандартна для визуального RL. В робототехнике, беспилотных аппаратах, игровых симуляторах — везде одна и та же проблема: агент должен принимать решения по неполной информации.

ViZDoom — это API к DOOM (1993), адаптированный для машинного обучения @kempka2016vizdoom. Платформа даёт псевдотрёхмерное окружение с противниками, несколькими картами, динамическим освещением и — что важно — со скоростью симуляции десятки тысяч кадров в секунду. За сутки на современном GPU можно набрать больше 800 миллионов наблюдений. Это делает ViZDoom удобным полигоном для экспериментов. В 2025-2026 годах вышли работы вроде NitroGen @magne2025nitrogen и SIMA 2 @bolton2025sima2, где агенты ориентируются в 3D-пространстве по картинке. Но вопрос выбора архитектуры памяти для RL-агента остаётся открытым. GRU @cho2014gru, Transformer @vaswani2017attention, Mamba-2 @dao2024transformers и Perceiver IO @jaegle2022perceiver дают разный баланс между качеством политики, скоростью работы и потреблением памяти. Систематического сравнения всех четырёх в одинаковых условиях в литературе нет.

Цель, стало быть, одна: взять единый алгоритм (APPO), единую среду (doom_benchmark) и одинаковый бюджет обучения, после чего сравнить архитектуры. Для этого требуется подключить ViZDoom к Sample Factory и организовать интерфейс смены модуля памяти, реализовать Mamba2Core, TransformerCore и PerceiverCore, провести пилотные запуски на 10 млн шагов для проверки сходимости, затем обучить каждую архитектуру на 50 млн шагов (с тремя seed для ключевых конфигураций) и сопоставить их по награде, FPS, пиковому VRAM и скорости, с которой награда выходит на плато.

Объект — нейросетевые архитектуры, обрабатывающие последовательности наблюдений; предмет — их применение в визуальной навигации при частичной наблюдаемости.

Новизна сводится к тому, что четыре архитектуры ставятся в идентичные условия @smirnov2025rlbenchnet: один и тот же алгоритм, одни и те же предобработка и бюджет, а не каждая со своими гиперпараметрами и окружением. Основное внимание уделено Mamba-2 — селективным SSM с линейной сложностью, которые в последнее время позиционируются как альтернатива трансформерам @gu2023mamba.

= Исследование предметной области

== Обучение с подкреплением в условиях частичной наблюдаемости

=== Постановка задачи POMDP

Обычный MDP: агент видит состояние $s_t$ целиком, выбирает действие $a_t$, получает награду. Вся информация доступна.

В визуальных средах агент видит только картинку $o_t = O(s_t)$. По одному кадру не понять, где он и что вокруг. Это POMDP — частично наблюдаемый процесс. Чтобы принимать адекватные решения, агент должен помнить историю: $(o_1, a_1, ..., o_t)$.

Без памяти политика $pi(a_t | o_t)$ слепа к контексту. Агент реагирует на текущий кадр и не помнит, что делал шаг назад. Память накапливает наблюдения, сжимает их в скрытое состояние, обновляет на каждом шаге. Как устроена эта память — главный вопрос работы.

=== RL-алгоритмы для визуальной навигации

PPO — стандарт @schulman2017ppo. Клиппирование не даёт политике меняться рывками, стабильно и просто. SAC — то же, но с энтропийной регуляризацией: агент поощряется за разведку. Это иногда полезно, но SAC требует больше памяти из-за replay buffer.

Decision Mamba: сделать SSM самим RL-алгоритмом, без отдельной памяти. Идея интересная, но на старте работы она не была проверена на ViZDoom. Рискованно.

APPO из Sample Factory @petrenko2020samplefactory был выбран по следующим причинам: (1) асинхронный сбор опыта — GPU не простаивает; (2) ModelCore позволяет вставить любую архитектуру памяти, не меняя цикл обучения; (3) для doom_benchmark уже были baseline от авторов фреймворка.

#figure(
  table(
    inset: 8pt,
    columns: 4,
    [Критерий], [PPO], [SAC], [APPO],
    [Тип], [on-policy], [off-policy], [on-policy],
    [Стабильность], [Высокая], [Средняя], [Высокая],
    [Параллелизм], [Нет], [Нет], [Асинхронный],
    [Буфер опыта], [Нет], [Replay buffer], [Нет],
    [VRAM], [Умеренное], [Высокое], [Умеренное],
  ),
  caption: [Сравнение RL-алгоритмов],
) <tab-rl-algo>

=== ViZDoom как исследовательская платформа

ViZDoom — API к DOOM для ML @kempka2016vizdoom. Сценарии: лабиринты (my_way_home), бои (defend_the_center, doom_benchmark), хилка (health_gathering). Каждый настраивается конфигом.

Частичная наблюдаемость тут не искусственная, а естественная: вид от первого лица, узкий угол обзора. Никаких радаров. Ранние работы @hafner2016deep, @mnih2015dqn продемонстрировали, что DQN способен обучаться непосредственно с пикселей в DOOM, но качество политики напрямую зависело от того, как агент учитывал историю наблюдений.

В работе используется doom_benchmark. Во-первых, под него есть baseline от авторов Sample Factory @petrenko2020samplefactory. Во-вторых, это стандартный сценарий для сравнения с литературой.

== Архитектуры памяти для обработки последовательностей

=== GRU — рекуррентный baseline

Стандартный рекуррентный слой, вокруг которого строится сравнение. GRU @cho2014gru — это LSTM @hochreiter1997lstm без отдельной ячейки памяти; два вентиля (обновления и сброса) вместо трёх, параметров на треть меньше, а способность моделировать временные зависимости практически та же. В RL GRU прижился из-за предсказуемости: градиенты не взрываются, скрытое состояние 512 элементов занимает 2 КБ на среду, тысяча параллельных агентов не создаёт проблем с памятью. BPTT, recurrence=32. В Sample Factory GRU стоит по умолчанию @petrenko2020samplefactory, и он же выступает в качестве baseline.

=== Трансформеры: внимание без затухания

Self-attention @vaswani2017attention принципиально лишён проблемы затухающих градиентов — любые две позиции соединяются напрямую. Цена — $O(L^2)$ по длине последовательности: контекст больше 128 шагов уже не помещается в VRAM, а если горизонт планирования короткий, выигрыша от внимания нет, хотя платить за квадратичную сложность всё равно приходится. Трансформеры чувствительны к гиперпараметрам и требуют много данных; для doom_benchmark с окном 64 шага они, как показали эксперименты, не дают приемлемого качества.

=== Mamba-2: селективные SSM

Mamba-2 @dao2024transformers основана на State Space Duality — по сути, линейная рекуррента, параметры которой зависят от входного сигнала. В отличие от старых SSM (S4, S5), где динамика фиксирована после обучения, селективность позволяет модели самой решать, что запомнить, а что отбросить. Сложность — $O(L)$, на длинных последовательностях Mamba-2 быстрее трансформера. Архитектурно: дискретная свёртка + селективное SSM + triton-ядра. Предшественник Mamba @gu2023mamba показал конкурентоспособные результаты на языке при линейном инференсе, но на ViZDoom систематически не проверялся. Если Mamba-2 даёт качество трансформера при линейной сложности, она может заменить и GRU, и трансформеры — это и проверяется в работе. Отметим Spatial-Mamba @xiao2024spatialmamba — адаптацию SSM для изображений, свидетельствующую, что селективные модели работают не только с текстом.

=== Perceiver IO: отделение входа от вычислений

Perceiver IO @jaegle2022perceiver (DeepMind) устроен иначе: размер входных данных не привязан к вычислительной стоимости. Модель оперирует фиксированным латентным массивом (256–512 векторов), а вход любой длины проецируется в него через кросс-внимание; самовнимание затем обрабатывает массив внутри. Для визуального RL это могло бы быть удобно — картинка 640×480 после энкодера даёт сотни признаков, и Perceiver обрабатывает их с постоянной стоимостью. На практике, однако, настройка оказалась сложной: размер массива, число итераций, способ инициализации латентного пространства — всё влияет на результат, и для doom_benchmark с фиксированным размером наблюдения преимущество переменного входа не реализуется.

=== Обоснование выбора архитектур для сравнения

Четыре класса: GRU (рекуррентные), Transformer (внимание), Mamba-2 (SSM), Perceiver IO (кросс-внимание). Сравниваем по трём осям: качество, скорость, память.

== Sample Factory: асинхронный фреймворк для RL

=== Архитектура APPO

Sample Factory @petrenko2020samplefactory — RL-фреймворк, заточенный на скорость. Policy workers собирают опыт параллельно и кидают в очередь. Leaner забирает и обновляет веса.

APPO: workers не ждут leaner-а — работают с последней версией политики, какая есть. Leaner копит траектории и обновляется, когда хватает данных. Схема хорошо грузит GPU: 8 workers × 8 сред = 64 параллельных окружения, 10-20K FPS. Для ViZDoom это стандарт.

=== Интерфейс ModelCore и интеграция кастомных моделей

ModelCore: два метода. forward(head_output, rnn_states) — на вход выход энкодера и скрытое состояние, на выход новый выход + новое состояние. core_output_size — размерность для декодера.

Любой класс, реализующий ModelCore, подключается без изменения остального кода. PackedSequence пакует траектории разной длины в один тензор — градиенты не перетекают между эпизодами.

== Постановка задачи исследования

Как четыре архитектуры — GRU, Transformer, Mamba-2, Perceiver IO — сравниваются по качеству, скорости и памяти в ViZDoom при одинаковых условиях? Насколько Mamba-2 приближается к трансформеру по качеству, сохраняя линейную сложность?

= Проектирование и разработка

== Архитектура программного комплекса

Архитектура строится по принципу разделения уровней: среда (ViZDoom), тренировочный цикл (Sample Factory / APPO) и модель политики (PolicyModel). Внутри модели модуль памяти вынесен в отдельный сменный компонент ModelCore — между свёрточным энкодером и декодером политики. Энкодер и декодер фиксированы, меняется только середина, что позволяет тестировать каждую архитектуру изолированно в идентичных условиях.

На рис. <fig-arch> приведена детальная архитектура PolicyModel: CNNEncoder, сменный модуль памяти (с четырьмя реализациями), и декодер. Пунктирной линией показан поток rnn_states, которым управляет Sample Factory.

#figure(
  image("architecture_diagram.svg", width: 100%),
  caption: [Архитектура PolicyModel со сменным модулем памяти. Показаны четыре реализации ModelCore: GRU (встроенный baseline sample-factory), Mamba-2, Transformer и Perceiver IO (кастомные). CNNEncoder и PolicyHead/ValueHead фиксированы. Снизу — контекст интеграции в тренировочный цикл Sample Factory],
) <fig-arch>

== Реализация Mamba2Core

Главная проблема — у Mamba-2 два внутренних состояния: conv_state и ssm_state. Они должны жить между шагами эпизода и умирать на его границе. Sample Factory умеет управлять ровно одним rnn_states — автоматом обнуляет его при конце эпизода. Значит, надо упаковать два массива Mamba-2 в один тензор.

=== Управление состоянием инференса

Mamba2StateEncoder сериализует conv_state и ssm_state в плоский тензор, совместимый с rnn_states. На forward распаковывает обратно.

Граница эпизода ловится по L2-норме: если меньше 1e-6 — считаем, что состояние обнулили, инициализируем свежее. Почему порог, а не точный ноль? Потому что через очередь между процессами летят числа с плавающей точкой — там погрешности неизбежны.

Листинг 1 — Mamba2StateEncoder: кодирование и декодирование состояния:

```python
class Mamba2StateEncoder:
    """Encodes/decodes Mamba-2 internal state (conv_state + ssm_state)
    to/from a flat tensor that fits in rnn_states."""

    def __init__(self, d_ssm: int, d_conv_dim: int, nheads: int,
                 headdim: int, d_state: int, d_conv: int):
        self.conv_state_size = d_conv_dim * d_conv
        self.ssm_state_size = nheads * headdim * d_state
        self.total_size = self.conv_state_size + self.ssm_state_size

    def encode(self, conv_state, ssm_state) -> torch.Tensor:
        """Flatten conv_state (batch, d_conv_dim, d_conv)
           and ssm_state (batch, nheads, headdim, d_state)
           into one tensor (batch, total_size)."""
        conv_flat = conv_state.reshape(conv_state.shape[0], -1)
        ssm_flat = ssm_state.reshape(ssm_state.shape[0], -1)
        return torch.cat([conv_flat, ssm_flat], dim=-1)

    def decode(self, flat_state, device=None, dtype=None):
        """Split flat tensor back into conv_state and ssm_state."""
        batch = flat_state.shape[0]
        conv_flat = flat_state[:, :self.conv_state_size]
        ssm_flat = flat_state[:, self.conv_state_size:]
        conv_state = conv_flat.reshape(batch, self.d_conv_dim, self.d_conv)
        ssm_state = ssm_flat.reshape(batch, self.nheads, self.headdim, self.d_state)
        return conv_state, ssm_state
```

=== Интеграция с ModelCore sample-factory

Метод forward принимает head_output (PackedSequence на обучении, тензор на инференсе) и rnn_states, возвращает выход и новое состояние. На обучении PackedSequence распаковывается в 3D-тензор, проходит через Mamba-2 блоки, пакуется обратно. На инференсе создаётся InferenceParams, состояние загружается из rnn_states, после прохода кодируется обратно.

Листинг 2 — Mamba2Core.forward: ключевой метод интеграции:

```python
class Mamba2Core(ModelCore):
    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if is_seq:
            # Training: PackedSequence -> (T, B, D)
            x_data = _unpack_packed_sequence_2d(head_output)
        else:
            # Inference: (B, D) -> (1, B, D)
            x_data = head_output.unsqueeze(0)

        # Mamba expects (B, T, D)
        x_data = x_data.permute(1, 0, 2)
        x_data = self.input_proj(x_data)

        if is_seq:
            # Training: process full sequence
            if getattr(self.cfg, 'gradient_checkpointing', True):
                x_data = self._forward_with_checkpointing(x_data)
            else:
                x_data = self.mamba_layers(x_data)
        else:
            # Inference: decode state, process one step, encode back
            batch_size = x_data.shape[0]
            inference_params = self._create_inference_params(batch_size)
            self._load_states_from_rnn(inference_params, rnn_states)
            inference_params.seqlen_offset = 1

            for norm, wrapped in zip(self.mamba_norms, self.mamba_wrapped):
                x_data = wrapped(norm(x_data),
                               inference_params=inference_params)

            new_rnn_states = self._get_inference_states(inference_params)

        x_data = x_data.permute(1, 0, 2)  # back to (T, B, D)

        if is_seq:
            x = _pack_to_2d_sequence(x_data, head_output)
            new_rnn_states = rnn_states
        else:
            x = x_data.squeeze(0)

        return x, new_rnn_states
```

=== Gradient checkpointing

Промежуточные активации не хранятся, а пересчитываются на обратном проходе. Минус 50% VRAM, плюс 20-30% времени. Включается gradient_checkpointing = True, каждый блок обёрнут в torch.utils.checkpoint.checkpoint.

Листинг 3 — Gradient checkpointing в Mamba2Core:

```python
def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint

    def segment_forward(x, norm, mamba_layer):
        return mamba_layer(norm(x))

    output = x
    for norm, wrapped in zip(self.mamba_norms, self.mamba_wrapped):
        output = checkpoint(
            segment_forward,
            output, norm, wrapped,
            use_reentrant=False,
        )
    return output
```

=== Регистрация через фабрику

Mamba2Factory — picklable-обёртка, которая регистрируется в глобальной фабрике моделей sample-factory. Это необходимо, так как sample-factory порождает workers через multiprocessing, и фабрика должна сериализоваться.

Листинг 4 — Регистрация Mamba-2 в sample-factory:

```python
class Mamba2Factory:
    __slots__ = ('_num_layers',)
    def __init__(self, num_layers):
        self._num_layers = num_layers
    def __call__(self, cfg, core_input_size):
        if cfg.use_rnn and cfg.rnn_type == 'mamba2':
            cfg.rnn_num_layers = self._num_layers
            return Mamba2Core(cfg, core_input_size)
        return default_make_core_func(cfg, core_input_size)

def register_mamba2(cfg=None):
    num_layers = cfg.rnn_num_layers if cfg is not None else 1
    factory = Mamba2Factory(num_layers)
    global_model_factory().register_model_core_factory(factory)
```

После вызова register_mamba2(cfg) достаточно указать rnn_type='mamba2' в конфиге — sample-factory сам создаст Mamba2Core при инициализации workers.

=== Параметры конфигурации

Модель настраивается группой mamba:
- mamba_d_model = 512 — размерность скрытого пространства
- mamba_d_state = 128 — размерность состояния SSM
- mamba_d_conv = 4 — ядро свёртки
- mamba_expand = 1 — расширение
- mamba_headdim = 128 — размерность головы
- mamba_ngroups = 1 — число групп

Значения — из HPO (раздел 2.5). Подогнаны под 512, как у GRU, для честного сравнения.

== Реализация TransformerCore

Стек трансформеров с каузальной маской @vaswani2017attention. 2-4 слоя, pre-norm, dropout, 512 скрытых, 8 голов. Окно контекста — 64 шага (дальше не лезет по VRAM).

На инференсе состояние кодируется как скользящее окно последних 64 токенов. Когда поступает новый кадр, окно сдвигается, и трансформер обрабатывает всё окно заново. Это дороже Mamba-2 ($O(L^2)$ против $O(L)$), но даёт доступ ко всей истории внутри окна.

Листинг 5 — TransformerCore: sliding window и каузальная маска:

```python
class TransformerCore(ModelCore):
    def __init__(self, cfg, input_size):
        self.d_model = getattr(cfg, 'transformer_d_model', cfg.rnn_size)
        self.nhead = getattr(cfg, 'transformer_nhead', 8)
        self.num_layers = getattr(cfg, 'transformer_num_layers', cfg.rnn_num_layers)
        self.window_size = getattr(cfg, 'transformer_window_size', 64)
        self.dim_feedforward = getattr(cfg, 'transformer_dim_feedforward', self.d_model * 4)

        self.state_encoder = TransformerStateEncoder(self.d_model, self.window_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True,
            ))

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.window_size, self.d_model) * 0.02)
        self.core_output_size = self.d_model

    def forward(self, head_output, rnn_states):
        # unpack, project -> (B, T, D)
        x_data = _unpack_packed_sequence_2d(head_output).permute(1, 0, 2)
        x_data = self.input_proj(x_data)
        B, T, D = x_data.shape

        if T > 1:  # Training: causal mask
            x_data = x_data + self.pos_embedding[:, :T, :]
            mask = _causal_mask(T, x_data.device)
            for norm, layer in zip(self.norms, self.layers):
                x_data = layer(x_data, src_mask=mask)
            out = x_data[:, -1:, :]
            new_rnn_states = rnn_states
        else:  # Inference: sliding window
            buffer = self.state_encoder.decode(rnn_states, B)
            buffer = torch.roll(buffer, shifts=-1, dims=1)
            buffer[:, -1:, :] = x_data + self.pos_embedding[:, -1:, :]
            mask = _causal_mask(self.window_size, x_data.device)
            for norm, layer in zip(self.norms, self.layers):
                buffer = layer(buffer, src_mask=mask)
            out = buffer[:, -1:, :]
            new_rnn_states = self.state_encoder.encode(buffer.detach())

        out = out.permute(1, 0, 2)
        x = _pack_to_2d_sequence(out.expand(T, -1, -1).contiguous(), head_output)
        return x, new_rnn_states
```

== Реализация PerceiverCore

Вход проецируется на латентный массив через кросс-внимание @jaegle2022perceiver, массив жуётся трансформерными блоками. Размер массива настраивается. Начальная конфигурация: 32 латентных вектора размерности 512. На инференсе латентный массив кодируется в rnn_states — это позволяет sample-factory управлять его жизненным циклом.

Листинг 6 — PerceiverCore: кросс-внимание и латентный массив:

```python
class PerceiverCore(ModelCore):
    def __init__(self, cfg, input_size):
        self.num_latents = getattr(cfg, 'perceiver_num_latents', 32)
        self.d_latents = getattr(cfg, 'perceiver_d_latents', 512)
        self.num_blocks = getattr(cfg, 'perceiver_num_blocks', 2)

        self.state_encoder = PerceiverStateEncoder(self.num_latents, self.d_latents)
        self.input_proj = nn.Linear(input_size, self.d_model)
        self.latent = nn.Parameter(torch.randn(1, self.num_latents, self.d_latents) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, self.d_model) * 0.02)

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(self.d_latents, self.d_model, num_heads=8)
            for _ in range(self.num_blocks)
        ])
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(self.d_latents, num_heads=8)
            for _ in range(self.num_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_latents), nn.Linear(self.d_latents, self.d_model))
        self.core_output_size = self.d_model

    def forward(self, head_output, rnn_states):
        x_data = _unpack_packed_sequence_2d(head_output).permute(1, 0, 2)
        x_data = self.input_proj(x_data)
        B, T, D = x_data.shape

        if T > 1:  # Training
            x_data = x_data + self.pos_encoding[:, :T, :]
            latent = self.latent.expand(B, -1, -1)
            for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
                latent = cross(latent, x_data)
                latent = self_attn(latent)
            out = self.output_proj(latent.mean(dim=1)).unsqueeze(1)
            new_rnn_states = rnn_states
        else:  # Inference
            latent = self.latent.expand(B, -1, -1).detach()
            if rnn_states.norm().item() > 1e-6:
                latent = self.state_encoder.decode(rnn_states, B)
            x_data = x_data + self.pos_encoding[:, :1, :]
            latent = self.cross_blocks[0](latent, x_data)
            latent = self.self_blocks[0](latent)
            out = self.output_proj(latent.mean(dim=1)).unsqueeze(1)
            new_rnn_states = self.state_encoder.encode(latent.detach())

        out = out.permute(1, 0, 2)
        x = _pack_to_2d_sequence(out.expand(T, -1, -1).contiguous(), head_output)
        return x, new_rnn_states
```

== Система HPO на базе Optuna

=== Байесовская оптимизация Mamba-2

Optuna, TPE-семплер, ASHA-прунинг после 10 эпох. Пространство: d_model 256/512/1024, d_state 64/128, headdim 64/128, lr 1e-5..1e-3, weight_decay 0..0.1. Проведено 85 trials по 50M шагов, из которых 37 успешно завершились (54% отказов из-за OOM/timeout).

Лучший trial #62: d_model=512, d_state=128, headdim=128, lr=4.05e-4, weight_decay=0.00196, expand=1. Mean reward 13.34, within-run std 3.14, max 16.94. Оптимизация заняла 42 минуты (2507 секунд на GPU). Все лучшие trial используют AdamW с lr~4e-4 и weight_decay~0.002.

#figure(
  table(
    inset: 8pt,
    columns: 4,
    [Параметр], [Диапазон поиска], [Лучшее значение], [Влияние],
    [$d_("model")$], [256 / 512 / 1024], [512], [Высокое],
    [$d_("state")$], [64 / 128], [128], [Среднее],
    [headdim], [64 / 128], [128], [Низкое],
    [lr], [$10^(-5)..10^(-3)$], [$4.05 times 10^(-4)$], [Высокое],
    [weight_decay], [$0..0.1$], [$1.96 times 10^(-3)$], [Среднее],
    [expand], [1 / 2], [1], [Низкое],
  ),
  caption: [Пространство поиска HPO и результаты лучшего trial #62 — Mamba-2],
) <tab-hpo>

=== HPO для Transformer

Проведено 15 trials для TransformerCore (то же пространство, добавлены window_size 32/64/128, dim_feedforward 1024/2048, dropout 0.1/0.2). Все 15 успешно завершены — Transformer стабилен и не требует больших ресурсов.

Лучший trial #5: best_reward=2.03 (d_model=512, nhead=8, window_size=64, dim_feedforward=1024, dropout=0.2, lr=8.36e-4, AdamW). Прирост относительно конфигурации по умолчанию (1.01) — всего 2×. Это подтверждает, что окно контекста 64 шага — фундаментальное ограничение: трансформер не видит дальше этого горизонта, а для doom_benchmark необходимо помнить существенно более длинные зависимости. Увеличение window_size до 128-256 — предмет дальнейших экспериментов.

=== HPO для Perceiver IO

HPO запущен (10 trials по 25M шагов, ~49 минут каждый). На момент написания завершён trial #0: best_reward=1.67 (d_model=512, latents=16, d_latents=256, blocks=1, heads=8, lr=3.9e-4, AdamW), что ниже конфигурации по умолчанию (1.97). Perceiver IO требует существенно больше времени на подбор гиперпараметров — результатов раньше 6-8 trials ожидать не стоит.

== Экспериментальный стенд

Стенд: Intel Ultra 9 285H, RTX 5090 Laptop GPU (24 GB), Linux. Программное обеспечение: Sample Factory 2.1.1, PyTorch 2.11.0+cu130, ViZDoom 1.3.0, CUDA 13.0, cuDNN 9.1.9, Python 3.13.12.

Конфигурация фиксирована: APPO, doom_benchmark, 8 workers × 8 сред, batch 4096, rollout 64, recurrence 32, hidden 512. Наблюдения: чёрно-белые кадры 640×480, FRAME_SKIP=4, стек из 4 последних кадров на вход энкодера. Награда: дельта счётчика убийств за шаг (стандартная для doom_benchmark). Seed по умолчанию — 0; для seed=42 применялся аргумент командной строки (явно задокументировано в конфигах запусков). config.json сохраняется в train_dir для каждого запуска — для воспроизводимости. Замеры FPS выполнены утилитой sample-factory (среднее за всё обучение).

= Экспериментальное исследование

== Методология сравнения

=== Метрики

Средняя награда за эпизод — главная метрика. Ещё FPS, пиковый VRAM, скорость сходимости.

=== Протокол

Каждая архитектура обучалась на 50M шагов. Для GRU (baseline и оптимизированный) и Mamba-2 проведено по 3-4 seed для оценки разброса; для Mamba-1, Transformer и Perceiver IO — по одному seed (при низкой абсолютной награде разброс между seed несущественен). Дополнительно выполнен запуск GRU на 250M шагов, чтобы оценить потолок рекуррентной архитектуры.

== Пилотные эксперименты

=== HPO: поиск гиперпараметров Mamba-2

85 trials по 50M шагов. d_model и learning rate влияют больше всего. d_model = 512 стабильно выигрывает у 256 и 1024 — видимо, золотая середина между выразительностью и скоростью.

=== Анализ чувствительности

Лучший lr = 4.05e-4 — в 4 раза выше, чем у GRU (1e-4). SSM не страдают взрывом градиентов, можно гнать быстрее. weight_decay = 0.00196 оказался критичен: без него награда начинает прыгать на поздних шагах.

== Результаты полного обучения

GRU со стоковыми гиперпараметрами Sample Factory дал по трём seed: 12.43, 14.37 и 13.95 — средняя лучшая награда 13.58 ± 1.02. Разброс между seed порядка 15% типичен для on-policy алгоритмов; within-run std по последним 20% итераций составил 0.33–0.70. FPS варьируется от 24 000 до 56 000 (зависит от загрузки системы), GPU загружен на 65%.

Когда к GRU применили гиперпараметры, найденные HPO для Mamba-2 (AdamW, lr=4.05e-4, weight_decay=0.00196, exploration_loss_coeff=0.002), результаты изменились кардинально: seed1 — 17.44, seed2 — 16.00, seed3 — 20.15, средняя — 17.86 ± 2.11. Прирост относительно baseline — +31%, причём seed3 (20.15) почти дотягивается до результата GRU 250M (22.45) при пятикратной экономии вычислительных ресурсов. Стандартные гиперпараметры Sample Factory для doom_benchmark, выходит, далеки от оптимальных — настройка важна не меньше, чем выбор архитектуры.

Mamba-2 с конфигурацией trial #62 (d_model=512, d_state=128, headdim=128, lr=4.05e-4) запускали на 4 seed. Результаты: 14.87, 14.84, 19.10, 15.97 — средняя 16.20 ± 2.01, максимальная 19.10. Высокий across-seed std (2.01) и within-run std (0.61–1.28) — следствие природы среды: награда в doom_benchmark скачет от эпизода к эпизоду в зависимости от поведения противника, и это не дефект архитектуры. По сходимости Mamba-2 стабильна, но к гиперпараметрам чувствительна: без HPO качество падает на 20–30%.

Mamba-1 — предшественник с неселективным SSM — получил 12.59 (seed1), на 22% ниже Mamba-2 HPO, при FPS 37 133. Селективность, то есть зависимость параметров динамики от входа, даёт ощутимый выигрыш.

Трансформер со sliding window 64, 2 слоями и 8 головами дал best_reward 1.01 (seed1). HPO из 15 trials поднял планку до 2.03 — это всё равно на порядок ниже GRU baseline. Окно 64 шага — фундаментальное ограничение: для doom_benchmark нужно помнить зависимости на всём rollout, а трансформер за пределы окна не выходит. Увеличение до 256–512 шагов упёрлось бы в VRAM из-за $O(L^2)$.

Perceiver IO (32 латентных вектора размерности 512) показал best_reward 1.97 — на порядок ниже GRU. HPO запущен, лучший на момент написания — 1.67. Для doom_benchmark с фиксированным размером наблюдения Perceiver избыточен; его преимущества (переменный вход) не востребованы.

Долгий прогон GRU на 250M шагов подтвердил плато: best_reward 22.45, финальная награда 20.95, средняя по последним 20% итераций — 20.28 ± 0.66. После 150M шагов рост прекращается; FPS 56 455 — снижения скорости нет.

Потребление видеопамяти core-модулями замерялось отдельно. Результаты приведены в табл. @tab-vram. Полная же модель, как видно из сводной таблицы далее, расходует от 5 до 9 GB: энкодер захватывает львиную долю.

=== Сводная таблица

#figure(
  table(
    inset: 8pt,
    columns: 7,
    [Архитектура], [Seed], [Meanᵃ], [Stdᵇ], [Max], [FPS], [VRAM],
    [GRU baseline], [3], [13.58], [1.02], [14.37], [24–56K], [~6 GB],
    [GRU + Opt.HP], [3], [17.86], [2.11], [20.15], [17–25K], [~6 GB],
    [Mamba-2 HPO], [4], [16.20], [2.01], [19.10], [15–23K], [~7 GB],
    [Mamba-1], [1], [12.59], [—], [12.59], [37K], [~6 GB],
    [Transformer], [1], [1.01], [—], [2.03ᶜ], [29K], [~6 GB],
    [Perceiver IO], [1], [1.97], [—], [1.97], [8K], [~8 GB],
    [GRU 250M], [1], [20.28ᵈ], [0.66ᵈ], [22.45], [56K], [~6 GB],
  ),
  caption: [
    Сводные результаты экспериментов. ᵃMean — средняя лучшая награда по seed (best reward за обучение).
    ᵇStd — across-seed стандартное отклонение (разброс между seed).
    ᶜПосле HPO — 2.03 (недостаточно для практического применения).
    ᵈGRU 250M: средняя и std по последним 20% итераций.
  ],
) <tab-results>

== Анализ результатов

=== Кривые обучения

#figure(
  image("reward_comparison.png", width: 100%),
  caption: [Сравнение кривых обучения всех архитектур на 50M шагов. Ось X — фреймы, ось Y — средняя награда за эпизод (EMA 0.99). Для архитектур с несколькими seed показан seed с максимальной наградой. Mamba-1, Transformer и Perceiver IO значительно уступают GRU и Mamba-2],
) <fig-reward>

#figure(
  image("gru_3seed_comparison.png", width: 100%),
  caption: [GRU baseline против GRU с гиперпараметрами Mamba-2 (3 seed). Ось X — фреймы, ось Y — средняя награда за эпизод. Оптимизированная GRU стабильно опережает baseline на всех seed. Seed3 достигает 20.15],
) <fig-gru-3s>

#figure(
  image("gru_250m_curve.png", width: 100%),
  caption: [GRU 250M — долгий прогон. Плато после ~150M шагов, финальная награда 20.95, максимальная 22.45],
) <fig-gru-250>

#figure(
  image("mamba2_seed_variance.png", width: 100%),
  caption: [Mamba-2 HPO: разброс между 4 seed. Seed3 показывает 19.10, остальные — 14.8–16.0. Высокая вариативность характерна для on-policy RL],
) <fig-mamba-var>

GRU с гиперпараметрами Mamba-2 растёт быстрее остальных: 55% финальной награды набирается за первые 10M (рис. <fig-reward>). Mamba-2 идёт плавнее, хотя разброс между seed выше. Плато GRU 250M, как видно на рис. <fig-gru-250>, наступает около 150M. Mamba-1, Transformer и Perceiver IO уступают значительно — их кривые обучения не поднимаются выше 12–13 даже за 50M шагов.

Если расположить архитектуры по средней лучшей награде за 50M, порядок получается такой: GRU с оптимизированными HP (17.86 ± 2.11, seed3 — 20.15), затем Mamba-2 HPO (16.20 ± 2.01, seed3 — 19.10), GRU baseline (13.58 ± 1.02), Mamba-1 (12.59), Perceiver IO (1.97) и Transformer (1.01). GRU 250M (22.45) здесь вне конкурса, но ценой пятикратного увеличения шагов.

По гиперпараметрам: GRU терпит lr от 1e-4 до 4e-4 почти без изменения результата. Mamba-2, напротив, — шаг в сторону от оптимума, и награда падает на 20–30%. Perceiver IO капризен к размеру латентного массива (уменьшение с 32 до 16 не дало прироста), а Transformer принципиально ограничен окном контекста — HPO бесполезна, пока window_size не увеличится хотя бы до 256.

= Заключение

В работе сравнивались четыре архитектуры памяти для RL — GRU, Mamba-2, Transformer, Perceiver IO. Для этого собран стенд на ViZDoom и Sample Factory, где модуль памяти подключается через интерфейс ModelCore и может быть заменён без затрагивания энкодера или декодера политики. Реализованы Mamba2Core (с двухкомпонентным состоянием conv_state + ssm_state, сериализацией в rnn_states и gradient checkpointing), TransformerCore со скользящим окном и каузальной маской, PerceiverCore с латентным массивом и кросс-вниманием. Проведена байесовская оптимизация для Mamba-2 (85 trials, 54% отказов из-за OOM) и для Transformer (15 trials). Эксперименты выполнены на 50M шагов для всех архитектур (3–4 seed для ключевых конфигураций плюс GRU 250M для оценки потолка).

GRU с гиперпараметрами Mamba-2 (AdamW, lr=4e-4) даёт 17.86 ± 2.11 против 13.58 ± 1.02 у стокового GRU — прирост +31% только за счёт настройки. Сама Mamba-2 (HPO best) достигает 16.20 ± 2.01, то есть на 11% ниже, но при линейной сложности $O(L)$ против квадратичной у трансформера. GRU 250M упёрся в 22.45, после 150M шагов роста нет. Mamba-1 (12.59), Transformer (1.01) и Perceiver IO (1.97) существенно уступают — первые две архитектуры не дают приемлемого качества в данной постановке.

GRU остаётся самым надёжным выбором для on-policy визуального RL: он нетребователен к ресурсам, прощает неточную настройку гиперпараметров и предсказуемо выходит на плато после 150M шагов. Оптимизация гиперпараметров даёт прирост, сопоставимый со сменой архитектуры — это важно само по себе. Mamba-2 — перспективная альтернатива: качество, близкое к GRU, при линейной сложности инференса, но без тщательной настройки HP её потенциал не раскрывается.

Из сравнения видно, что выбор архитектуры упирается в доступные ресурсы. При ограниченном бюджете (до 50M шагов, без HPO) GRU со стандартными настройками Sample Factory выдаёт 13–14 и не требует настройки. Если время на HPO есть, тот же GRU с AdamW и lr~4e-4 поднимается до 17–20 — лучший показатель среди всех архитектур при 50M. Mamba-2 интересна там, где важна скорость инференса на длинных последовательностях (сотни шагов): её $O(L)$ даёт 16–19, но без HPO она уступает стоковому GRU. Transformer и Perceiver IO для doom_benchmark не подходят: они не достигают приемлемого качества при фиксированном размере наблюдения и без кардинального увеличения окна контекста.

Дальнейшие направления включают увеличение окна контекста Transformer до 256+ шагов (чтобы проверить, решит ли это проблему), гибридные VLA-модели вроде CombatVLA @chen2025combatvla, где Mamba-2 может выступать в роли backbone, и ablation studies для оценки вклада gradient checkpointing и размерности состояния SSM в общую производительность.

#bibliography("references.bib")

#show: appendixes

= Приложение А. Графики обучения

#figure(
  image("reward_comparison.png", width: 100%),
  caption: [Рис. А.1 — Сравнение кривых обучения всех архитектур на 50M шагов. Ось X — количество собранных фреймов, ось Y — средняя награда за эпизод],
)

#figure(
  image("gru_3seed_comparison.png", width: 100%),
  caption: [Рис. А.2 — GRU baseline против GRU с гиперпараметрами Mamba-2 (3 seed)],
)

#figure(
  image("gru_250m_curve.png", width: 100%),
  caption: [Рис. А.3 — GRU 250M. Плато после ~150M, максимальная награда 22.45],
)

#figure(
  image("mamba2_seed_variance.png", width: 100%),
  caption: [Рис. А.4 — Mamba-2 HPO: разброс между 4 seed],
)

#figure(
  image("pilot_comparison.png", width: 100%),
  caption: [Рис. А.5 — Пилотные эксперименты GRU и Mamba-2 на 10M шагов],
)

Результаты полного обучения:

#figure(
  table(
    inset: 8pt,
    columns: 7,
    [Архитектура], [Seed], [Mean], [Std], [Max], [FPS], [VRAM],
    [GRU baseline], [3], [13.58], [1.02], [14.37], [24–56K], [~6 GB],
    [GRU + Opt.HP], [3], [17.86], [2.11], [20.15], [17–25K], [~6 GB],
    [Mamba-2 HPO], [4], [16.20], [2.01], [19.10], [15–23K], [~7 GB],
    [Mamba-1], [1], [12.59], [—], [12.59], [37K], [~6 GB],
    [Transformer], [1], [1.01], [—], [2.03ᵃ], [29K], [~6 GB],
    [Perceiver IO], [1], [1.97], [—], [1.97], [8K], [~8 GB],
    [GRU 250M], [1], [20.28ᵇ], [0.66ᵇ], [22.45], [56K], [~6 GB],
  ),
  caption: [Таблица А.1 — Сводные результаты экспериментов. ᵃПосле HPO (15 trials). ᵇСредняя и std по последним 20% итераций],
)

= Приложение Б. Конфигурации экспериментов

== Базовая конфигурация (APPO)

#figure(
  table(
    inset: 8pt,
    columns: 2,
    [Параметр], [Значение],
    [algorithm], [APPO],
    [env], [doom_benchmark],
    [num_workers], [8],
    [num_envs_per_worker], [8],
    [batch_size], [4096],
    [rollout], [64],
    [recurrence], [32],
    [hidden_size], [512],
    [rnn_num_layers], [1],
    [learning_rate], [1e-4 (GRU) / 4.05e-4 (Mamba-2)],
    [optimizer], [AdamW],
    [weight_decay], [0 (GRU) / 1.96e-3 (Mamba-2)],
    [gamma], [0.99],
    [gae_lambda], [0.95],
    [value_loss_coeff], [1.0],
    [exploration_loss_coeff], [0.002 (Mamba-2)],
    [grad_norm], [4.0],
    [num_epochs], [1],
    [num_minibatches], [1],
  ),
  caption: [Базовая конфигурация APPO],
)

== Конфигурация Mamba-2 (HPO best trial #62)

#figure(
  table(
    inset: 8pt,
    columns: 2,
    [Параметр], [Значение],
    [mamba_d_model], [512],
    [mamba_d_state], [128],
    [mamba_headdim], [128],
    [mamba_expand], [1],
    [mamba_d_conv], [4],
    [mamba_ngroups], [1],
    [gradient_checkpointing], [True],
    [rnn_type], [mamba2],
    [learning_rate], [4.05e-4],
    [weight_decay], [1.96e-3],
    [optimizer], [AdamW],
    [exploration_loss_coeff], [0.002],
  ),
  caption: [Конфигурация Mamba-2, trial #62],
)

== Конфигурация TransformerCore

#figure(
  table(
    inset: 8pt,
    columns: 2,
    [Параметр], [Значение],
    [transformer_d_model], [512],
    [transformer_nhead], [8],
    [transformer_num_layers], [2],
    [transformer_window_size], [64],
    [transformer_dim_feedforward], [2048],
    [transformer_dropout], [0.1],
    [rnn_type], [transformer],
  ),
  caption: [Конфигурация TransformerCore],
)

== Конфигурация PerceiverCore

#figure(
  table(
    inset: 8pt,
    columns: 2,
    [Параметр], [Значение],
    [perceiver_num_latents], [32],
    [perceiver_d_latents], [512],
    [perceiver_num_blocks], [2],
    [perceiver_num_heads], [8],
    [perceiver_dropout], [0.1],
    [rnn_type], [perceiver],
  ),
  caption: [Конфигурация PerceiverCore],
)

== Версии программного обеспечения

#figure(
  table(
    inset: 8pt,
    columns: 2,
    [Компонент], [Версия],
    [PyTorch], [2.11.0+cu130],
    [Sample Factory], [2.1.1],
    [ViZDoom], [1.3.0],
    [CUDA], [13.0],
    [cuDNN], [9.1.9],
    [Python], [3.13.12],
  ),
  caption: [Версии программного обеспечения экспериментального стенда],
)

== Потребление видеопамяти (core-only)

В табл. @tab-vram сведены результаты замеров core-модулей (энкодер и декодер в расчёт не брались). Число параметров и потребление видеопамяти не всегда коррелируют напрямую: Mamba-2 с 0.9M параметров даёт 0.40 GB, а Perceiver IO, у которого параметров больше в 16 раз, — всего 0.42 GB. SSM-сканирование вынуждено хранить conv_state и ssm_state на каждом шаге, а латентный массив Perceiver IO фиксирован (32×512) и не растёт с длиной последовательности. Полная модель даёт 5–9 GB, основная доля уходит на свёрточный энкодер с его 4096 кадрами.

#figure(
  table(
    inset: 8pt,
    columns: 3,
    [Архитектура], [Параметры (core)], [VRAM (core)],
    [GRU], [1.6M], [0.07 GB],
    [Mamba-2], [0.9M], [0.40 GB],
    [Mamba-1], [1.7M], [0.15 GB],
    [Transformer], [3.2M], [0.20 GB],
    [Perceiver IO], [14.5M], [0.42 GB],
  ),
  caption: [Потребление видеопамяти core-only (без учёта энкодера и декодера)],
) <tab-vram>
