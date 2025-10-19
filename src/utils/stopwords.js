/**
 * Анти-шейминг фильтр токсичных слов
 * MVP: простые regex, без ML
 */

const STOPWORDS_PATTERNS = [
  // Вес и тело
  /жирн(ая|ую|ой|ый|ые)/gi,
  /толст(ая|ую|ой|ый|ые)/gi,
  /жиртрест/gi,
  /сало/gi,
  /корова/gi,
  /свинья/gi,
  /кит/gi,
  /слон(иха)?/gi,
  /бегемот/gi,
  /тумба/gi,
  /бочка/gi,
  /шкаф/gi,
  
  // Призывы похудеть
  /похудей/gi,
  /сбрось вес/gi,
  /иди в зал/gi,
  /перестань жрать/gi,
  /меньше ешь/gi,
  /на диету/gi,
  
  // Общая токсичность
  /уродин(а|ка)/gi,
  /мерзость/gi,
  /противн(ая|о)/gi,
  /отвратительн(ая|о)/gi,
  
  // Сексуальные домогательства
  /шлюх(а|и)/gi,
  /проститутка/gi,
  /б(л|ля)(д|т)(ь|и|ь|ъ)/gi,
  
  // Очевидный спам
  /эскорт/gi,
  /проверенная/gi,
  /досуг/gi,
  /интим за/gi,
  /массаж.*выезд/gi,
];

/**
 * Проверяет текст на наличие стоп-слов
 * @param {string} text - текст для проверки
 * @returns {{isClean: boolean, matches: string[]}}
 */
export function checkStopwords(text) {
  if (!text || typeof text !== 'string') {
    return { isClean: true, matches: [] };
  }

  const matches = [];
  
  for (const pattern of STOPWORDS_PATTERNS) {
    const found = text.match(pattern);
    if (found) {
      matches.push(...found);
    }
  }

  return {
    isClean: matches.length === 0,
    matches: [...new Set(matches)], // unique
  };
}

/**
 * Очищает текст от стоп-слов (заменяет на ***)
 * @param {string} text
 * @returns {string}
 */
export function sanitizeText(text) {
  if (!text) return text;
  
  let sanitized = text;
  
  for (const pattern of STOPWORDS_PATTERNS) {
    sanitized = sanitized.replace(pattern, '***');
  }
  
  return sanitized;
}
