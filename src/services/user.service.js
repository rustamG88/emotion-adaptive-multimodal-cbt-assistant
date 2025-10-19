import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

/**
 * User service - работа с профилями пользователей
 */

export class UserService {
  /**
   * Находит или создаёт пользователя по Telegram ID
   */
  static async findOrCreate(tgId, userData = {}) {
    let user = await prisma.user.findUnique({
      where: { tgId: String(tgId) },
    });

    if (!user) {
      user = await prisma.user.create({
        data: {
          tgId: String(tgId),
          username: userData.username,
          firstName: userData.firstName,
          role: 'unknown',
          age: 18,
          city: 'unknown',
          goals: JSON.stringify([]),
          interests: JSON.stringify([]),
          photos: JSON.stringify([]),
        },
      });
    }

    return user;
  }

  /**
   * Получить профиль по Telegram ID
   */
  static async getByTgId(tgId) {
    return await prisma.user.findUnique({
      where: { tgId: String(tgId) },
    });
  }

  /**
   * Обновить профиль
   */
  static async update(tgId, data) {
    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data: {
        ...data,
        updatedAt: new Date(),
      },
    });
  }

  /**
   * Обновить lastOnline
   */
  static async updateOnline(tgId) {
    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data: { lastOnline: new Date() },
    });
  }

  /**
   * Проверка, заполнен ли профиль (готов к поиску)
   */
  static isProfileComplete(user) {
    const photos = JSON.parse(user.photos || '[]');
    const interests = JSON.parse(user.interests || '[]');
    
    return (
      user.role !== 'unknown' &&
      user.age >= 18 &&
      user.city !== 'unknown' &&
      photos.length > 0 &&
      user.bio &&
      interests.length > 0
    );
  }

  /**
   * Проверка Premium статуса
   */
  static isPremium(user) {
    if (!user.premiumUntil) return false;
    return new Date(user.premiumUntil) > new Date();
  }

  /**
   * Активировать Premium
   */
  static async activatePremium(tgId, durationDays = 30) {
    const premiumUntil = new Date();
    premiumUntil.setDate(premiumUntil.getDate() + durationDays);

    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data: { premiumUntil },
    });
  }

  /**
   * Забанить пользователя
   */
  static async ban(tgId, reason, durationHours = null) {
    const data = {
      isBanned: true,
      bannedReason: reason,
    };

    if (durationHours) {
      const bannedUntil = new Date();
      bannedUntil.setHours(bannedUntil.getHours() + durationHours);
      data.bannedUntil = bannedUntil;
    }

    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data,
    });
  }

  /**
   * Разбанить
   */
  static async unban(tgId) {
    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data: {
        isBanned: false,
        bannedUntil: null,
      },
    });
  }

  /**
   * Удалить профиль (soft delete)
   */
  static async softDelete(tgId) {
    return await prisma.user.update({
      where: { tgId: String(tgId) },
      data: {
        isDeleted: true,
        deletedAt: new Date(),
      },
    });
  }
}
