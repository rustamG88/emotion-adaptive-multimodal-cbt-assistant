import { Bot, session } from 'grammy';
import { hydrate } from '@grammyjs/hydrate';
import { conversations, createConversation } from '@grammyjs/conversations';
import { logger } from './utils/logger.js';

// TODO: Import handlers
// import { startHandler } from './bot/handlers/start.js';
// import { onboardingConversation } from './bot/conversations/onboarding.js';

export async function createBot() {
  const token = process.env.BOT_TOKEN;
  
  if (!token) {
    throw new Error('BOT_TOKEN is not defined in environment variables');
  }

  const bot = new Bot(token);

  // Hydration for easier API calls
  bot.use(hydrate());

  // Session for FSM
  bot.use(session({
    initial: () => ({
      state: 'idle',
      payload: {},
    }),
  }));

  // Conversations plugin for FSM
  bot.use(conversations());

  // TODO: Register conversations
  // bot.use(createConversation(onboardingConversation));

  // Error handling
  bot.catch((err) => {
    const ctx = err.ctx;
    logger.error({ 
      err: err.error,
      update: ctx.update,
      user: ctx.from?.id,
    }, 'Error in bot handler');
  });

  // TODO: Register handlers
  // bot.command('start', startHandler);
  
  // Fallback handler
  bot.on('message', async (ctx) => {
    await ctx.reply('Привет! 👋 Бот в разработке. Скоро запустимся!');
  });

  return bot;
}
