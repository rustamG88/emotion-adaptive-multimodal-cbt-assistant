import 'dotenv/config';
import { createBot } from './bot.js';
import { logger } from './utils/logger.js';

async function main() {
  try {
    const bot = await createBot();
    
    // Graceful shutdown
    process.once('SIGINT', () => {
      logger.info('Received SIGINT, stopping bot...');
      bot.stop();
    });
    process.once('SIGTERM', () => {
      logger.info('Received SIGTERM, stopping bot...');
      bot.stop();
    });

    // Start bot
    await bot.start();
    logger.info('✅ Bot started successfully');
    
  } catch (error) {
    logger.error({ err: error }, '❌ Failed to start bot');
    process.exit(1);
  }
}

main();
