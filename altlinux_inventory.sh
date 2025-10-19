#!/bin/bash

################################################################################
# Скрипт инвентаризации системы ALT Linux 10.4
# Собирает информацию о hardware и формирует отчет для инвентаризации
################################################################################

set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Проверка прав суперпользователя
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ОШИБКА]${NC} Скрипт должен быть запущен с правами root (sudo)"
   exit 1
fi

# Определение путей
HOSTNAME=$(hostname)
CURRENT_DIR=$(dirname "$(realpath "$0")")
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="${CURRENT_DIR}/${HOSTNAME}_${TIMESTAMP}.log"
CSV_FILE="${CURRENT_DIR}/${HOSTNAME}_${TIMESTAMP}.csv"

# Проверка обязательных команд
REQUIRED_CMDS="uname hostname ip awk"
for cmd in $REQUIRED_CMDS; do
    if ! command -v "$cmd" &> /dev/null; then
        echo -e "${RED}[ОШИБКА]${NC} Команда '$cmd' не найдена"
        exit 1
    fi
done

################################################################################
# Вспомогательные функции
################################################################################

# Функция для безопасного выполнения команд
safe_exec() {
    "$@" 2>/dev/null || echo "Не доступно"
}

# Функция для получения значения из dmidecode
get_dmidecode() {
    local type="$1"
    local field="$2"
    if command -v dmidecode &> /dev/null; then
        dmidecode -t "$type" 2>/dev/null | grep -m1 "^\s*${field}:" | cut -d: -f2- | xargs || echo "Не определено"
    else
        echo "Не определено"
    fi
}

# Функция для форматированного вывода
print_section() {
    local title="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    printf "║ %-74s ║\n" "$title" | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
}

# Функция для форматированного вывода поля
print_field() {
    local name="$1"
    local value="$2"
    printf "  %-30s : %s\n" "$name" "$value" | tee -a "$LOG_FILE"
}

################################################################################
# Начало сбора информации
################################################################################

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          СБОР ИНФОРМАЦИИ ДЛЯ ИНВЕНТАРИЗАЦИИ ALT LINUX 10.4                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Создание заголовка лог-файла
{
echo "════════════════════════════════════════════════════════════════════════════"
echo "                  ОТЧЕТ ИНВЕНТАРИЗАЦИИ СИСТЕМЫ"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Дата и время сбора: $(date '+%d.%m.%Y %H:%M:%S')"
echo "Имя хоста: $HOSTNAME"
echo ""
} > "$LOG_FILE"

################################################################################
# 1. ОБЩАЯ ИНФОРМАЦИЯ О СИСТЕМЕ
################################################################################

echo -e "${BLUE}[1/10]${NC} Сбор общей информации о системе..."

print_section "1. ОБЩАЯ ИНФОРМАЦИЯ О СИСТЕМЕ"

# Операционная система
if [ -f /etc/altlinux-release ]; then
    OS_VERSION=$(cat /etc/altlinux-release)
elif [ -f /etc/os-release ]; then
    OS_VERSION=$(grep "^PRETTY_NAME=" /etc/os-release | cut -d'"' -f2)
else
    OS_VERSION="ALT Linux (версия не определена)"
fi
print_field "Операционная система" "$OS_VERSION"

# Версия ядра
KERNEL_VERSION=$(uname -r)
print_field "Версия ядра" "$KERNEL_VERSION"

# Архитектура
ARCH=$(uname -m)
print_field "Архитектура" "$ARCH"

# Uptime
UPTIME=$(uptime -p | sed 's/up //')
print_field "Время работы" "$UPTIME"

################################################################################
# 2. ИНФОРМАЦИЯ О КОМПЬЮТЕРЕ/СЕРВЕРЕ
################################################################################

echo -e "${BLUE}[2/10]${NC} Сбор информации о системе..."

print_section "2. ИНФОРМАЦИЯ О КОМПЬЮТЕРЕ"

SYSTEM_MANUFACTURER=$(get_dmidecode "system" "Manufacturer")
SYSTEM_PRODUCT=$(get_dmidecode "system" "Product Name")
SYSTEM_VERSION=$(get_dmidecode "system" "Version")
SYSTEM_SERIAL=$(get_dmidecode "system" "Serial Number")
SYSTEM_UUID=$(get_dmidecode "system" "UUID")

print_field "Производитель" "$SYSTEM_MANUFACTURER"
print_field "Модель" "$SYSTEM_PRODUCT"
print_field "Версия" "$SYSTEM_VERSION"
print_field "Серийный номер" "$SYSTEM_SERIAL"
print_field "UUID" "$SYSTEM_UUID"

################################################################################
# 3. ИНФОРМАЦИЯ О BIOS/UEFI
################################################################################

echo -e "${BLUE}[3/10]${NC} Сбор информации о BIOS..."

print_section "3. ИНФОРМАЦИЯ О BIOS/UEFI"

BIOS_VENDOR=$(get_dmidecode "bios" "Vendor")
BIOS_VERSION=$(get_dmidecode "bios" "Version")
BIOS_DATE=$(get_dmidecode "bios" "Release Date")

print_field "Производитель BIOS" "$BIOS_VENDOR"
print_field "Версия BIOS" "$BIOS_VERSION"
print_field "Дата выпуска BIOS" "$BIOS_DATE"

################################################################################
# 4. ИНФОРМАЦИЯ О МАТЕРИНСКОЙ ПЛАТЕ
################################################################################

echo -e "${BLUE}[4/10]${NC} Сбор информации о материнской плате..."

print_section "4. ИНФОРМАЦИЯ О МАТЕРИНСКОЙ ПЛАТЕ"

MB_MANUFACTURER=$(get_dmidecode "baseboard" "Manufacturer")
MB_PRODUCT=$(get_dmidecode "baseboard" "Product Name")
MB_VERSION=$(get_dmidecode "baseboard" "Version")
MB_SERIAL=$(get_dmidecode "baseboard" "Serial Number")

print_field "Производитель" "$MB_MANUFACTURER"
print_field "Модель" "$MB_PRODUCT"
print_field "Версия" "$MB_VERSION"
print_field "Серийный номер" "$MB_SERIAL"

################################################################################
# 5. ИНФОРМАЦИЯ О ПРОЦЕССОРЕ
################################################################################

echo -e "${BLUE}[5/10]${NC} Сбор информации о процессоре..."

print_section "5. ИНФОРМАЦИЯ О ПРОЦЕССОРЕ"

if command -v lscpu &> /dev/null; then
    CPU_MODEL=$(lscpu | grep "^Model name:" | cut -d: -f2- | xargs)
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    CPU_THREADS=$(lscpu | grep "^Thread" | awk '{print $4}')
    CPU_FREQ=$(lscpu | grep "CPU max MHz" | awk '{printf "%.2f GHz", $4/1000}')
    [ -z "$CPU_FREQ" ] && CPU_FREQ=$(lscpu | grep "CPU MHz" | awk '{printf "%.2f GHz", $3/1000}')
else
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2- | xargs)
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)
    CPU_THREADS="Не определено"
    CPU_FREQ="Не определено"
fi

print_field "Модель процессора" "$CPU_MODEL"
print_field "Количество ядер" "$CPU_CORES"
print_field "Потоков" "$CPU_THREADS"
print_field "Частота" "$CPU_FREQ"

################################################################################
# 6. ИНФОРМАЦИЯ О ПАМЯТИ
################################################################################

echo -e "${BLUE}[6/10]${NC} Сбор информации о памяти..."

print_section "6. ИНФОРМАЦИЯ О ОПЕРАТИВНОЙ ПАМЯТИ"

if command -v free &> /dev/null; then
    RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
else
    RAM_TOTAL=$(awk '/MemTotal/ {printf "%.1f GB", $2/1024/1024}' /proc/meminfo)
fi

print_field "Общий объем RAM" "$RAM_TOTAL"

# Информация о модулях памяти
if command -v dmidecode &> /dev/null; then
    echo "" | tee -a "$LOG_FILE"
    echo "  Установленные модули памяти:" | tee -a "$LOG_FILE"
    dmidecode -t memory 2>/dev/null | grep -A 5 "Memory Device" | grep -E "Size:|Speed:|Manufacturer:|Part Number:|Serial Number:" | while read line; do
        if [[ $line =~ "Size:" ]] && [[ ! $line =~ "No Module" ]]; then
            echo "    ────────────────────────────────" | tee -a "$LOG_FILE"
            echo "    $line" | tee -a "$LOG_FILE"
        elif [[ ! $line =~ "Size:" ]] && [[ ! -z "$line" ]]; then
            echo "    $line" | tee -a "$LOG_FILE"
        fi
    done
fi

################################################################################
# 7. ИНФОРМАЦИЯ О ДИСКАХ
################################################################################

echo -e "${BLUE}[7/10]${NC} Сбор информации о дисках..."

print_section "7. ИНФОРМАЦИЯ О НАКОПИТЕЛЯХ"

if command -v lsblk &> /dev/null; then
    echo "" | tee -a "$LOG_FILE"
    
    # Получаем список физических дисков
    for disk in $(lsblk -dno NAME | grep -E '^(sd|nvme|vd|hd)'); do
        echo "  ┌─── Диск: /dev/$disk ───────────────────────────────────" | tee -a "$LOG_FILE"
        
        # Размер диска
        DISK_SIZE=$(lsblk -dno SIZE /dev/$disk)
        print_field "  │ Размер" "$DISK_SIZE"
        
        # Модель диска
        if [ -f /sys/block/$disk/device/model ]; then
            DISK_MODEL=$(cat /sys/block/$disk/device/model | xargs)
            print_field "  │ Модель" "$DISK_MODEL"
        fi
        
        # SMART информация
        if command -v smartctl &> /dev/null; then
            DISK_SERIAL=$(smartctl -i /dev/$disk 2>/dev/null | grep "Serial Number:" | cut -d: -f2- | xargs)
            DISK_FW=$(smartctl -i /dev/$disk 2>/dev/null | grep "Firmware Version:" | cut -d: -f2- | xargs)
            [ -n "$DISK_SERIAL" ] && print_field "  │ Серийный номер" "$DISK_SERIAL"
            [ -n "$DISK_FW" ] && print_field "  │ Версия прошивки" "$DISK_FW"
        fi
        
        echo "  └──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
fi

################################################################################
# 8. СЕТЕВАЯ ИНФОРМАЦИЯ
################################################################################

echo -e "${BLUE}[8/10]${NC} Сбор сетевой информации..."

print_section "8. СЕТЕВАЯ ИНФОРМАЦИЯ"

# IP адреса
echo "  IP адреса:" | tee -a "$LOG_FILE"
ip -4 -o addr show | awk '{print $2, $4}' | while read iface addr; do
    if [[ "$iface" != "lo" ]]; then
        printf "    %-20s : %s\n" "$iface" "$addr" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"

# MAC адреса
echo "  MAC адреса:" | tee -a "$LOG_FILE"
ip -o link show | awk '{print $2, $17}' | sed 's/:$//' | while read iface mac; do
    if [[ "$iface" != "lo:" ]] && [[ "$mac" != "00:00:00:00:00:00" ]] && [[ -n "$mac" ]]; then
        printf "    %-20s : %s\n" "$iface" "$mac" | tee -a "$LOG_FILE"
    fi
done

################################################################################
# 9. ПОЛЬЗОВАТЕЛИ СИСТЕМЫ
################################################################################

echo -e "${BLUE}[9/10]${NC} Сбор информации о пользователях..."

print_section "9. ПОЛЬЗОВАТЕЛИ СИСТЕМЫ"

echo "  Локальные пользователи (UID >= 1000):" | tee -a "$LOG_FILE"
awk -F':' '$3 >= 1000 && $3 != 65534 {printf "    - %s (UID: %s)\n", $1, $3}' /etc/passwd | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "  Активные сеансы:" | tee -a "$LOG_FILE"
if command -v who &> /dev/null; then
    who | awk '{printf "    - %s на %s с %s\n", $1, $2, $3}' | tee -a "$LOG_FILE"
else
    echo "    Информация недоступна" | tee -a "$LOG_FILE"
fi

################################################################################
# 10. ДОПОЛНИТЕЛЬНОЕ ОБОРУДОВАНИЕ
################################################################################

echo -e "${BLUE}[10/10]${NC} Сбор информации о дополнительном оборудовании..."

print_section "10. ДОПОЛНИТЕЛЬНОЕ ОБОРУДОВАНИЕ"

# Мониторы (только если есть X-сервер)
if [ -n "${DISPLAY:-}" ] && command -v xrandr &> /dev/null; then
    echo "  Подключенные мониторы:" | tee -a "$LOG_FILE"
    xrandr --query 2>/dev/null | grep " connected" | while read line; do
        echo "    - $line" | tee -a "$LOG_FILE"
    done
else
    echo "  Мониторы: Графическая среда не обнаружена или не запущена" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# Принтеры
if command -v lpstat &> /dev/null; then
    echo "  Настроенные принтеры:" | tee -a "$LOG_FILE"
    PRINTERS=$(lpstat -v 2>/dev/null)
    if [ -n "$PRINTERS" ]; then
        echo "$PRINTERS" | while read line; do
            echo "    - $line" | tee -a "$LOG_FILE"
        done
    else
        echo "    Принтеры не настроены" | tee -a "$LOG_FILE"
    fi
else
    echo "  Принтеры: cups не установлен" | tee -a "$LOG_FILE"
fi

# Видеокарта
echo "" | tee -a "$LOG_FILE"
if command -v lspci &> /dev/null; then
    VGA_INFO=$(lspci | grep -i 'vga\|3d\|display' | head -1)
    if [ -n "$VGA_INFO" ]; then
        print_field "Видеокарта" "$VGA_INFO"
    fi
fi

################################################################################
# СОЗДАНИЕ CSV ДЛЯ ИМПОРТА В ТАБЛИЦУ
################################################################################

echo ""
echo -e "${BLUE}[CSV]${NC} Создание файла для импорта в таблицу..."

{
echo "Поле,Значение"
echo "Дата инвентаризации,$(date '+%d.%m.%Y %H:%M:%S')"
echo "Имя хоста,$HOSTNAME"
echo "Операционная система,$OS_VERSION"
echo "Версия ядра,$KERNEL_VERSION"
echo "Производитель системы,$SYSTEM_MANUFACTURER"
echo "Модель системы,$SYSTEM_PRODUCT"
echo "Серийный номер системы,$SYSTEM_SERIAL"
echo "Производитель МП,$MB_MANUFACTURER"
echo "Модель МП,$MB_PRODUCT"
echo "Серийный номер МП,$MB_SERIAL"
echo "BIOS версия,$BIOS_VERSION"
echo "BIOS дата,$BIOS_DATE"
echo "Процессор,$CPU_MODEL"
echo "Ядер процессора,$CPU_CORES"
echo "Объем RAM,$RAM_TOTAL"

# Добавляем информацию о дисках
DISK_NUM=1
for disk in $(lsblk -dno NAME 2>/dev/null | grep -E '^(sd|nvme|vd|hd)'); do
    DISK_SIZE=$(lsblk -dno SIZE /dev/$disk)
    DISK_MODEL=$(cat /sys/block/$disk/device/model 2>/dev/null | xargs || echo "Неизвестно")
    DISK_SERIAL=""
    if command -v smartctl &> /dev/null; then
        DISK_SERIAL=$(smartctl -i /dev/$disk 2>/dev/null | grep "Serial Number:" | cut -d: -f2- | xargs || echo "Неизвестно")
    fi
    echo "Диск ${DISK_NUM} модель,$DISK_MODEL"
    echo "Диск ${DISK_NUM} размер,$DISK_SIZE"
    echo "Диск ${DISK_NUM} серийный номер,$DISK_SERIAL"
    ((DISK_NUM++))
done

# IP адреса
IP_LIST=$(ip -4 -o addr show | awk '$2 != "lo" {print $4}' | tr '\n' '; ')
echo "IP адреса,$IP_LIST"

# MAC адреса
MAC_LIST=$(ip -o link show | awk '$2 != "lo:" && $17 != "00:00:00:00:00:00" {print $17}' | tr '\n' '; ')
echo "MAC адреса,$MAC_LIST"

} > "$CSV_FILE"

################################################################################
# ЗАВЕРШЕНИЕ
################################################################################

{
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "                    КОНЕЦ ОТЧЕТА"
echo "════════════════════════════════════════════════════════════════════════════"
} >> "$LOG_FILE"

# Установка прав доступа
chmod 0644 "$LOG_FILE"
chmod 0644 "$CSV_FILE"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     СБОР ИНФОРМАЦИИ ЗАВЕРШЕН                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓${NC} Отчет сохранен в:     ${YELLOW}$LOG_FILE${NC}"
echo -e "${GREEN}✓${NC} CSV для таблицы:      ${YELLOW}$CSV_FILE${NC}"
echo ""
echo -e "Для просмотра отчета: ${BLUE}cat \"$LOG_FILE\"${NC}"
echo -e "Для импорта в LibreOffice Calc откройте файл: ${BLUE}$CSV_FILE${NC}"
echo ""
