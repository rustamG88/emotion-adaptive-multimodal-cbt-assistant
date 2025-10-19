/**
 * Алгоритм скоринга профилей для поиска
 * MVP: простая линейная комбинация
 */

/**
 * Рассчитывает score между двумя профилями
 * @param {Object} myProfile - профиль текущего пользователя
 * @param {Object} targetProfile - профиль кандидата
 * @returns {number} score (0-100)
 */
export function calculateScore(myProfile, targetProfile) {
  let score = 0;

  // 1. Совпадение целей (0-20 баллов)
  const myGoals = JSON.parse(myProfile.goals || '[]');
  const targetGoals = JSON.parse(targetProfile.goals || '[]');
  const goalsIntersection = myGoals.filter(g => targetGoals.includes(g));
  score += goalsIntersection.length * 5; // макс 20 если 4+ совпадения

  // 2. Общие интересы (0-25 баллов)
  const myInterests = JSON.parse(myProfile.interests || '[]');
  const targetInterests = JSON.parse(targetProfile.interests || '[]');
  const interestsIntersection = myInterests.filter(i => targetInterests.includes(i));
  score += Math.min(interestsIntersection.length * 5, 25);

  // 3. Верификация (0-20 баллов)
  if (targetProfile.verifiedAt) {
    score += 20;
  }

  // 4. Расстояние (0 до -20 баллов штраф)
  if (myProfile.lat && myProfile.lon && targetProfile.lat && targetProfile.lon) {
    const distance = calculateDistance(
      myProfile.lat, myProfile.lon,
      targetProfile.lat, targetProfile.lon
    );
    
    if (distance <= 10) {
      score += 10; // очень близко - бонус
    } else if (distance <= 50) {
      // нет штрафа
    } else if (distance <= 100) {
      score -= 5;
    } else {
      score -= 20;
    }
  }

  // 5. Активность (0-10 баллов)
  const hoursSinceOnline = (Date.now() - new Date(targetProfile.lastOnline).getTime()) / (1000 * 60 * 60);
  if (hoursSinceOnline < 1) {
    score += 10;
  } else if (hoursSinceOnline < 24) {
    score += 5;
  } else if (hoursSinceOnline < 72) {
    score += 2;
  }

  // 6. Premium (0-5 баллов)
  if (targetProfile.premiumUntil && new Date(targetProfile.premiumUntil) > new Date()) {
    score += 5;
  }

  // 7. Репутация (0 до -50 штраф)
  if (targetProfile.reportCount > 0) {
    score -= targetProfile.reportCount * 10;
  }

  return Math.max(0, Math.min(100, score));
}

/**
 * Haversine formula для расстояния между координатами
 * @returns {number} расстояние в км
 */
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // радиус Земли в км
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  
  const a = 
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function toRad(degrees) {
  return degrees * Math.PI / 180;
}

/**
 * Сортирует профили по score
 * @param {Object} myProfile
 * @param {Array} candidates
 * @returns {Array} отсортированные профили с score
 */
export function rankProfiles(myProfile, candidates) {
  return candidates
    .map(candidate => ({
      ...candidate,
      score: calculateScore(myProfile, candidate),
    }))
    .sort((a, b) => b.score - a.score);
}
