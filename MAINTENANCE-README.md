# Скрипт обслуживания системы ALT Linux

Этот скрипт автоматизирует обслуживание клиентских машин ALT Linux, включая обновление пакетов, ядра и планирование перезагрузки при необходимости.

## Что делает скрипт

1. **Очистка журналов systemd**
   - Ротация журналов
   - Удаление логов старше 1 дня
   - Ограничение размера до 100MB

2. **Обновление пакетов**
   - Очистка кэша APT
   - Обновление списков пакетов
   - Полное обновление системы (dist-upgrade)

3. **Обновление ядра**
   - Автоматическое обновление ядра
   - Проверка версии запущенного ядра
   - Определение необходимости перезагрузки

4. **Очистка старых ядер**
   - Автоматическое удаление неиспользуемых версий ядра

5. **Обслуживание CUPS**
   - Отмена всех заданий печати
   - Перезапуск службы CUPS
   - Очистка логов CUPS

6. **Планирование перезагрузки**
   - Если установлено новое ядро, планируется перезагрузка:
     - В 18:35 через `atd` (если доступен)
     - Через 5 минут через `shutdown` (резервный вариант)

## Использование

### Одна команда (текущий вариант)

Через SSH или в терминале:

```bash
bash -lc 'journalctl --rotate && journalctl --vacuum-time=1d && journalctl --vacuum-size=100M && apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y; update-kernel -y || true; running="$(uname -r)"; latest="$(ls -1 /lib/modules 2>/dev/null | sort -V | tail -n1)"; need_reboot=0; [ -n "$latest" ] && [ "$latest" != "$running" ] && need_reboot=1; remove-old-kernels -y || true; command -v cancel >/dev/null && cancel -a || true; systemctl try-restart cups.service || true; : > /var/log/cups/access_log 2>/dev/null || true; : > /var/log/cups/error_log 2>/dev/null || true; if [ "$need_reboot" -eq 1 ]; then (systemctl is-active --quiet atd && echo reboot | at 18:35) || shutdown -r +5 "Kernel updated; rebooting in 5 minutes"; else echo "[INFO] No reboot needed"; fi'
```

### Использование скрипта (рекомендуется)

1. Скопируйте скрипт на клиентскую машину:
   ```bash
   scp system-maintenance.sh user@client:/usr/local/bin/
   ```

2. Сделайте скрипт исполняемым:
   ```bash
   ssh user@client 'chmod +x /usr/local/bin/system-maintenance.sh'
   ```

3. Запустите через SSH:
   ```bash
   ssh user@client 'sudo /usr/local/bin/system-maintenance.sh'
   ```

   Или с bash -lc:
   ```bash
   ssh user@client 'sudo bash -lc /usr/local/bin/system-maintenance.sh'
   ```

### Автоматизация через cron

Для регулярного обслуживания добавьте в `/etc/cron.d/system-maintenance`:

```cron
# Обслуживание системы каждую среду в 3:00
0 3 * * 3 root /usr/local/bin/system-maintenance.sh >> /var/log/system-maintenance.log 2>&1
```

### Массовое выполнение на нескольких машинах

С помощью простого bash цикла:

```bash
for host in client1 client2 client3; do
    echo "=== Обслуживание $host ==="
    ssh root@$host '/usr/local/bin/system-maintenance.sh'
done
```

С помощью parallel (быстрее):

```bash
parallel -j 5 "ssh root@{} '/usr/local/bin/system-maintenance.sh'" ::: client1 client2 client3
```

С помощью ansible:

```yaml
- hosts: clients
  tasks:
    - name: Run system maintenance
      script: system-maintenance.sh
      become: yes
```

## Требования

- **ОС**: ALT Linux
- **Права**: root или sudo
- **Команды**: 
  - `journalctl`
  - `apt-get`
  - `update-kernel`
  - `remove-old-kernels`
  - `systemctl`
  - `cancel` (опционально, для CUPS)
  - `at` (опционально, для планирования перезагрузки)

## Логирование

Для сохранения логов запускайте скрипт с перенаправлением:

```bash
/usr/local/bin/system-maintenance.sh 2>&1 | tee -a /var/log/system-maintenance.log
```

Или добавьте в скрипт:

```bash
exec 1> >(tee -a /var/log/system-maintenance.log)
exec 2>&1
```

## Безопасность

- Скрипт требует прав root
- Рекомендуется протестировать на тестовой машине перед массовым развёртыванием
- Обратите внимание на время перезагрузки (18:35 или +5 минут)
- При необходимости измените параметры очистки журналов и время перезагрузки

## Настройка

Вы можете изменить следующие параметры в скрипте:

- **Время хранения журналов**: `--vacuum-time=1d` (по умолчанию 1 день)
- **Размер журналов**: `--vacuum-size=100M` (по умолчанию 100 МБ)
- **Время перезагрузки**: `at 18:35` (по умолчанию 18:35)
- **Задержка перезагрузки**: `+5` минут (резервный вариант)

## Преимущества использования скрипта

✅ Читаемость и поддерживаемость кода  
✅ Подробные комментарии на каждом шаге  
✅ Информативные сообщения о ходе выполнения  
✅ Легко модифицировать и расширять  
✅ Можно версионировать в git  
✅ Проще отладка при возникновении проблем  

## Поддержка

При возникновении проблем проверьте:
1. Логи системы: `journalctl -xe`
2. Лог скрипта (если настроено логирование)
3. Доступность необходимых команд
4. Права доступа к файлам и директориям
