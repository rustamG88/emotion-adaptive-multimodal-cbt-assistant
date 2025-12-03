#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Cleaner - Очистка сканированных PDF от артефактов
Сохраняет текст, печати и подписи, удаляя только грязь сканера.

Требования:
    pip install opencv-python pdf2image Pillow numpy

Для Windows также нужен Poppler:
    https://github.com/osber/poppler-windows/releases
"""

import os
from pathlib import Path
import threading
from typing import Callable, Optional, List, Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ----- НАСТРОЙКА POPPLER ДЛЯ WINDOWS -----
POPPLER_PATH = r"C:\poppler-25.11.0\Library\bin"

# ----- ПАРАМЕТРЫ ОЧИСТКИ -----
class CleaningParams:
    """Параметры алгоритма очистки."""
    # Минимальная площадь объекта для сохранения (в пикселях)
    MIN_CONTENT_AREA = 50
    # Максимальная площадь шума для удаления
    MAX_NOISE_AREA = 30
    # Порог для определения полос (длина / ширина)
    STRIPE_ASPECT_RATIO = 15
    # Сила денойзинга (меньше = сохраняет больше деталей)
    DENOISE_STRENGTH = 8
    # Радиус inpaint для заполнения удалённых полос
    INPAINT_RADIUS = 5


# ---------- Алгоритм очистки ----------

def detect_colored_regions(img_rgb: np.ndarray) -> np.ndarray:
    """
    Находит цветные области (печати, подписи) — обычно синие/красные/фиолетовые.
    Возвращает маску цветных областей.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Синие оттенки (печати, подписи)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Красные оттенки (печати) - красный в HSV разбит на два диапазона
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    
    # Фиолетовые/пурпурные оттенки
    purple_lower = np.array([130, 50, 50])
    purple_upper = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    
    # Объединяем все цветные области
    color_mask = blue_mask | red_mask | purple_mask
    
    # Расширяем маску, чтобы захватить полностью печати
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.dilate(color_mask, kernel, iterations=2)
    
    return color_mask


def detect_stripes(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Находит вертикальные и горизонтальные полосы от сканера.
    Возвращает две маски: вертикальных и горизонтальных полос.
    """
    inv = 255 - gray
    
    # Вертикальные полосы
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vert_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    _, vert_mask = cv2.threshold(vert_lines, 30, 255, cv2.THRESH_BINARY)
    
    # Горизонтальные полосы
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    horiz_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    _, horiz_mask = cv2.threshold(horiz_lines, 30, 255, cv2.THRESH_BINARY)
    
    return vert_mask, horiz_mask


def remove_small_noise(binary: np.ndarray, max_area: int = 30) -> np.ndarray:
    """
    Удаляет мелкие шумовые компоненты, сохраняя текст и значимые объекты.
    """
    # Инвертируем, чтобы текст был белым (для connectedComponents)
    inv = 255 - binary
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    
    # Создаём маску для удаления
    noise_mask = np.zeros(binary.shape, dtype=np.uint8)
    
    for i in range(1, num_labels):  # 0 — фон
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Удаляем только очень мелкие точки
        if area < max_area:
            # Проверяем, что это не часть буквы (не слишком вытянутое)
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect < 5:  # Не линия/штрих
                noise_mask[labels == i] = 255
    
    # Убираем шум из результата
    result = binary.copy()
    result[noise_mask == 255] = 255  # Делаем белым (фон)
    
    return result


def smart_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Умная бинаризация с сохранением текста разной интенсивности.
    Комбинирует глобальную и адаптивную бинаризацию.
    """
    # Лёгкое размытие для уменьшения шума
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Глобальная бинаризация Otsu
    _, global_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Адаптивная бинаризация для слабого текста
    adaptive_bin = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 10
    )
    
    # Комбинируем: берём пересечение (текст, который виден в обоих методах)
    # Это убирает ложные срабатывания адаптивной бинаризации
    combined = cv2.bitwise_or(global_bin, adaptive_bin)
    
    # Но предпочитаем глобальную для чётких областей
    # Используем глобальную там, где контраст высокий
    local_std = cv2.blur((gray.astype(float) - cv2.blur(gray, (21, 21)).astype(float))**2, (21, 21))
    high_contrast = local_std > 500
    
    result = adaptive_bin.copy()
    result[high_contrast] = global_bin[high_contrast]
    
    return result


def clean_page(img_rgb: np.ndarray, params: CleaningParams = None) -> np.ndarray:
    """
    Очищает страницу от артефактов сканирования.
    Сохраняет текст, печати, подписи.
    
    Args:
        img_rgb: Страница в формате RGB (numpy array)
        params: Параметры очистки
    
    Returns:
        Очищенное изображение в RGB
    """
    if params is None:
        params = CleaningParams()
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Находим цветные области (печати, подписи) — их не трогаем
    color_mask = detect_colored_regions(img_rgb)
    
    # 2. Находим и удаляем полосы от сканера
    vert_mask, horiz_mask = detect_stripes(gray)
    stripe_mask = vert_mask | horiz_mask
    
    # Исключаем цветные области из маски полос
    stripe_mask[color_mask > 0] = 0
    
    # Заполняем полосы через inpaint
    if np.any(stripe_mask):
        gray = cv2.inpaint(gray, stripe_mask, params.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
    # 3. Мягкий денойзинг (сохраняем детали текста)
    denoised = cv2.fastNlMeansDenoising(
        gray, None, 
        h=params.DENOISE_STRENGTH,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # 4. Умная бинаризация
    binary = smart_binarize(denoised)
    
    # 5. Удаляем мелкий шум, сохраняя текст
    cleaned = remove_small_noise(binary, params.MAX_NOISE_AREA)
    
    # 6. Восстанавливаем цветные элементы (печати, подписи)
    # Конвертируем результат обратно в RGB
    result_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    
    # Накладываем оригинальные цветные области
    color_regions = color_mask > 0
    if np.any(color_regions):
        # Берём оригинальные пиксели в цветных областях
        result_rgb[color_regions] = img_rgb[color_regions]
    
    return result_rgb


def clean_page_grayscale(img_rgb: np.ndarray, params: CleaningParams = None) -> np.ndarray:
    """
    Очищает страницу и возвращает чёрно-белый результат.
    Печати и подписи преобразуются в оттенки серого.
    """
    if params is None:
        params = CleaningParams()
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Находим цветные области
    color_mask = detect_colored_regions(img_rgb)
    
    # 2. Удаляем полосы
    vert_mask, horiz_mask = detect_stripes(gray)
    stripe_mask = vert_mask | horiz_mask
    stripe_mask[color_mask > 0] = 0
    
    if np.any(stripe_mask):
        gray = cv2.inpaint(gray, stripe_mask, params.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
    # 3. Денойзинг
    denoised = cv2.fastNlMeansDenoising(
        gray, None,
        h=params.DENOISE_STRENGTH,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # 4. Бинаризация
    binary = smart_binarize(denoised)
    
    # 5. Удаление шума
    cleaned = remove_small_noise(binary, params.MAX_NOISE_AREA)
    
    # 6. Для цветных областей (печати) используем бинаризацию оригинала
    if np.any(color_mask > 0):
        # Берём серый канал из оригинала для печатей
        orig_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, stamp_binary = cv2.threshold(orig_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned[color_mask > 0] = stamp_binary[color_mask > 0]
    
    return cleaned


def process_pdf(
    input_pdf: str,
    output_pdf: str,
    dpi: int = 300,
    keep_color: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> None:
    """
    Обрабатывает PDF файл.
    
    Args:
        input_pdf: Путь к входному PDF
        output_pdf: Путь для сохранения результата
        dpi: Разрешение для обработки
        keep_color: Сохранять цветные элементы (печати) или делать ч/б
        progress_callback: Функция обратного вызова (current, total, message)
    """
    if progress_callback:
        progress_callback(0, 0, "Загрузка PDF...")
    
    pages = convert_from_path(
        input_pdf,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )
    
    total_pages = len(pages)
    cleaned_pages: List[Image.Image] = []
    params = CleaningParams()
    
    for i, page in enumerate(pages):
        if progress_callback:
            progress_callback(i, total_pages, f"Обработка страницы {i + 1} из {total_pages}...")
        
        page_np = np.array(page)
        
        if keep_color:
            cleaned = clean_page(page_np, params)
            pil_cleaned = Image.fromarray(cleaned)
        else:
            cleaned = clean_page_grayscale(page_np, params)
            pil_cleaned = Image.fromarray(cleaned)
        
        cleaned_pages.append(pil_cleaned)
    
    if not cleaned_pages:
        raise RuntimeError("Не удалось обработать ни одной страницы PDF")
    
    if progress_callback:
        progress_callback(total_pages, total_pages, "Сохранение PDF...")
    
    first, *rest = cleaned_pages
    first.save(
        output_pdf,
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=dpi,
    )
    
    if progress_callback:
        progress_callback(total_pages, total_pages, "Готово!")


# ---------- GUI ----------

class PdfCleanerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("🧹 PDF Cleaner — Очистка сканов")
        self.geometry("1000x700")
        self.configure(bg="#2b2b2b")
        
        # Стиль
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()
        
        # Переменные
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Выберите PDF для очистки")
        self.keep_color_var = tk.BooleanVar(value=True)
        self.dpi_var = tk.IntVar(value=300)
        
        self.original_img_tk = None
        self.cleaned_img_tk = None
        self._cancel_flag = threading.Event()
        self._processing = False
        
        self._build_ui()
    
    def _configure_styles(self):
        """Настройка стилей виджетов."""
        bg_color = "#2b2b2b"
        fg_color = "#e0e0e0"
        accent = "#4a9eff"
        
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent)
        self.style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#90EE90")
        
        self.style.configure(
            "Accent.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(20, 10)
        )
        
        self.style.configure(
            "TProgressbar",
            troughcolor="#3c3c3c",
            background=accent,
            thickness=20
        )
    
    def _build_ui(self):
        """Строит интерфейс."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # === Верхняя панель с настройками ===
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Заголовок
        ttk.Label(
            settings_frame, 
            text="📄 Очистка сканированных PDF",
            style="Header.TLabel"
        ).pack(anchor="w")
        
        # Входной файл
        input_frame = ttk.Frame(settings_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Входной PDF:").pack(side=tk.LEFT)
        ttk.Entry(
            input_frame, 
            textvariable=self.input_path_var, 
            width=70
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(
            input_frame, 
            text="📂 Обзор...", 
            command=self.select_input
        ).pack(side=tk.LEFT)
        
        # Выходной файл
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Сохранить как:").pack(side=tk.LEFT)
        ttk.Entry(
            output_frame, 
            textvariable=self.output_path_var, 
            width=70
        ).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(
            output_frame, 
            text="💾 Изменить...", 
            command=self.select_output
        ).pack(side=tk.LEFT)
        
        # Опции
        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(
            options_frame,
            text="Сохранять цветные элементы (печати, подписи)",
            variable=self.keep_color_var
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(options_frame, text="DPI:").pack(side=tk.LEFT)
        dpi_combo = ttk.Combobox(
            options_frame,
            textvariable=self.dpi_var,
            values=[150, 200, 300, 400],
            width=6,
            state="readonly"
        )
        dpi_combo.pack(side=tk.LEFT, padx=5)
        
        # Кнопки действий
        buttons_frame = ttk.Frame(settings_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.clean_button = ttk.Button(
            buttons_frame,
            text="🧹 Очистить PDF",
            style="Accent.TButton",
            command=self.start_clean
        )
        self.clean_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.cancel_button = ttk.Button(
            buttons_frame,
            text="❌ Отмена",
            command=self.cancel_processing,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT)
        
        # Прогресс-бар
        progress_frame = ttk.Frame(settings_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            length=400
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="0%",
            width=10
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Статус
        self.status_label = ttk.Label(
            settings_frame,
            textvariable=self.status_var,
            style="Status.TLabel"
        )
        self.status_label.pack(anchor="w", pady=5)
        
        # === Зона предпросмотра ===
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель — До
        left_frame = ttk.Frame(preview_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(left_frame, text="📋 Оригинал", style="Header.TLabel").pack()
        
        self.canvas_before = tk.Canvas(left_frame, bg="#3c3c3c", highlightthickness=0)
        self.canvas_before.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Правая панель — После
        right_frame = ttk.Frame(preview_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(right_frame, text="✨ Результат", style="Header.TLabel").pack()
        
        self.canvas_after = tk.Canvas(right_frame, bg="#3c3c3c", highlightthickness=0)
        self.canvas_after.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # --------- Обработчики ---------
    
    def select_input(self):
        """Выбор входного файла."""
        path = filedialog.askopenfilename(
            title="Выберите PDF для очистки",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        
        self.input_path_var.set(path)
        
        # Автогенерация имени выхода
        in_path = Path(path)
        default_out = in_path.with_name(in_path.stem + "_clean.pdf")
        self.output_path_var.set(str(default_out))
        
        # Показываем предпросмотр
        self.status_var.set("Загрузка предпросмотра...")
        self.update_idletasks()
        
        threading.Thread(
            target=self._load_preview,
            args=(path,),
            daemon=True
        ).start()
    
    def select_output(self):
        """Выбор места сохранения."""
        initial_dir = ""
        initial_file = ""
        
        if self.output_path_var.get():
            p = Path(self.output_path_var.get())
            initial_dir = str(p.parent)
            initial_file = p.name
        
        path = filedialog.asksaveasfilename(
            title="Сохранить очищенный PDF как...",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            self.output_path_var.set(path)
    
    def _load_preview(self, pdf_path: str):
        """Загружает предпросмотр первой страницы (в отдельном потоке)."""
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=150,  # Низкое разрешение для быстрого предпросмотра
                first_page=1,
                last_page=1,
                poppler_path=POPPLER_PATH,
            )
        except Exception as e:
            self.after(0, lambda: self._show_error(f"Не удалось открыть PDF:\n{e}"))
            return
        
        if not pages:
            return
        
        page_np = np.array(pages[0])
        
        # Очищаем для предпросмотра
        if self.keep_color_var.get():
            cleaned = clean_page(page_np)
        else:
            cleaned = clean_page_grayscale(page_np)
            cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        # Обновляем UI в главном потоке
        self.after(0, lambda: self._update_preview(page_np, cleaned))
    
    def _update_preview(self, original: np.ndarray, cleaned: np.ndarray):
        """Обновляет предпросмотр (в главном потоке)."""
        self.update_idletasks()
        
        # Получаем размеры canvas
        canvas_w = self.canvas_before.winfo_width()
        canvas_h = self.canvas_before.winfo_height()
        
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 450, 500
        
        # Масштабируем изображения
        pil_before = Image.fromarray(original)
        pil_after = Image.fromarray(cleaned)
        
        pil_before.thumbnail((canvas_w - 10, canvas_h - 10), Image.LANCZOS)
        pil_after.thumbnail((canvas_w - 10, canvas_h - 10), Image.LANCZOS)
        
        self.original_img_tk = ImageTk.PhotoImage(pil_before)
        self.cleaned_img_tk = ImageTk.PhotoImage(pil_after)
        
        # Отображаем
        self.canvas_before.delete("all")
        self.canvas_after.delete("all")
        
        self.canvas_before.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.original_img_tk,
            anchor="center"
        )
        self.canvas_after.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.cleaned_img_tk,
            anchor="center"
        )
        
        self.status_var.set("Предпросмотр загружен. Нажмите 'Очистить PDF' для обработки.")
    
    def start_clean(self):
        """Запускает очистку PDF."""
        input_pdf = self.input_path_var.get().strip()
        output_pdf = self.output_path_var.get().strip()
        
        if not input_pdf:
            messagebox.showwarning("Внимание", "Сначала выберите входной PDF.")
            return
        
        if not os.path.exists(input_pdf):
            messagebox.showerror("Ошибка", "Файл не найден.")
            return
        
        if not output_pdf:
            messagebox.showwarning("Внимание", "Укажите куда сохранить результат.")
            return
        
        # Подтверждение перезаписи
        if os.path.exists(output_pdf):
            if not messagebox.askyesno(
                "Подтверждение",
                f"Файл уже существует:\n{output_pdf}\n\nПерезаписать?"
            ):
                return
        
        self._processing = True
        self._cancel_flag.clear()
        
        self.clean_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0%")
        
        threading.Thread(
            target=self._clean_thread,
            args=(input_pdf, output_pdf),
            daemon=True
        ).start()
    
    def cancel_processing(self):
        """Отменяет обработку."""
        self._cancel_flag.set()
        self.status_var.set("Отмена...")
    
    def _clean_thread(self, input_pdf: str, output_pdf: str):
        """Поток обработки PDF."""
        def progress_callback(current: int, total: int, message: str):
            if self._cancel_flag.is_set():
                raise InterruptedError("Обработка отменена пользователем")
            
            if total > 0:
                percent = int((current / total) * 100)
            else:
                percent = 0
            
            self.after(0, lambda: self._update_progress(percent, message))
        
        try:
            process_pdf(
                input_pdf,
                output_pdf,
                dpi=self.dpi_var.get(),
                keep_color=self.keep_color_var.get(),
                progress_callback=progress_callback
            )
        except InterruptedError:
            self.after(0, lambda: self._on_cancelled())
            return
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))
            return
        
        self.after(0, lambda: self._on_complete(output_pdf))
    
    def _update_progress(self, percent: int, message: str):
        """Обновляет прогресс-бар."""
        self.progress_bar["value"] = percent
        self.progress_label.config(text=f"{percent}%")
        self.status_var.set(message)
    
    def _on_complete(self, output_pdf: str):
        """Вызывается при успешном завершении."""
        self._processing = False
        self.clean_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_bar["value"] = 100
        self.progress_label.config(text="100%")
        self.status_var.set("✅ Готово!")
        
        messagebox.showinfo(
            "Готово!",
            f"Очищенный файл сохранён:\n{output_pdf}"
        )
    
    def _on_cancelled(self):
        """Вызывается при отмене."""
        self._processing = False
        self.clean_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.status_var.set("Обработка отменена")
    
    def _on_error(self, error_msg: str):
        """Вызывается при ошибке."""
        self._processing = False
        self.clean_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.status_var.set("❌ Ошибка")
        
        messagebox.showerror("Ошибка", f"Не удалось обработать PDF:\n{error_msg}")
    
    def _show_error(self, message: str):
        """Показывает ошибку."""
        self.status_var.set("Ошибка")
        messagebox.showerror("Ошибка", message)


def main():
    app = PdfCleanerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
