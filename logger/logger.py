import logging
import sys

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename="debug.log"
        )

logger = logging.getLogger(__name__)
