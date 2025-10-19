import { InlineKeyboard } from 'grammy';

/**
 * Главное меню бота
 */
export function mainMenuKeyboard() {
  return new InlineKeyboard()
    .text('🔎 Поиск', 'menu:search')
    .text('👤 Моя анкета', 'menu:profile')
    .row()
    .text('⭐ Премиум', 'menu:premium')
    .text('⚙️ Настройки', 'menu:settings');
}

/**
 * Меню профиля
 */
export function profileMenuKeyboard() {
  return new InlineKeyboard()
    .text('📸 Изменить фото', 'profile:edit_photos')
    .row()
    .text('✍️ Изменить текст', 'profile:edit_bio')
    .text('🎯 Интересы', 'profile:edit_interests')
    .row()
    .text('🛡️ Верификация', 'profile:verification')
    .row()
    .text('🗑 Удалить анкету', 'profile:delete')
    .row()
    .text('⬅️ Назад', 'menu:main');
}

/**
 * Карточка пользователя в поиске
 */
export function profileCardKeyboard(targetUserId) {
  return new InlineKeyboard()
    .text('💬 Написать', `chat:open:${targetUserId}`)
    .text('❤️ Интересно', `like:send:${targetUserId}`)
    .row()
    .text('👎 Пропустить', `search:next`)
    .text('⚠️ Пожаловаться', `report:start:${targetUserId}`)
    .row()
    .text('⬅️ К поиску', 'menu:search');
}

/**
 * Меню админа
 */
export function adminMenuKeyboard() {
  return new InlineKeyboard()
    .text('📊 Статистика', 'admin:stats')
    .text('🧹 Модерация', 'admin:moderation')
    .row()
    .text('💵 Платежи', 'admin:payments')
    .row()
    .text('⬅️ Выход', 'menu:main');
}

/**
 * Кнопки модерации
 */
export function moderationActionsKeyboard(targetUserId, reportId) {
  return new InlineKeyboard()
    .text('✅ Отклонить жалобу', `moderate:dismiss:${reportId}`)
    .row()
    .text('⚠️ Предупредить', `moderate:warn:${targetUserId}:${reportId}`)
    .row()
    .text('🔇 Мут 24ч', `moderate:mute:${targetUserId}:${reportId}`)
    .row()
    .text('⛔ Бан', `moderate:ban:${targetUserId}:${reportId}`)
    .row()
    .text('⬅️ Назад', 'admin:moderation');
}

/**
 * Подтверждение действия
 */
export function confirmKeyboard(action, targetId) {
  return new InlineKeyboard()
    .text('✅ Да, подтверждаю', `confirm:${action}:${targetId}`)
    .row()
    .text('❌ Отмена', 'menu:main');
}
