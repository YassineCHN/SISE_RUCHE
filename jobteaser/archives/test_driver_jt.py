from jobteaser.archives.utils_jt_base import (
    DriverPool,
    handle_cloudflare_challenge,
    accept_cookies,
    SEARCH_URL,
)
import time
from selenium.webdriver.common.by import By

driver = DriverPool(pool_size=1, headless=False)
webdriver = driver.get_driver()

webdriver.get(SEARCH_URL)
time.sleep(2.0)
handle_cloudflare_challenge(webdriver)
accept_cookies(webdriver)
time.sleep(0.2)
cards = webdriver.find_elements(By.CLASS_NAME, "JobAdCard_main__1mTeA")
print("Avant scroll : ", len(cards))
for _ in range(3):
    webdriver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(1)
time.sleep(2)
print("Après scroll : ", len(cards))
if len(cards) > 0:
    print(cards[0].text)
print(webdriver.execute_script("return document.body.className"))
print(webdriver.current_url)
print(webdriver.title)
print(webdriver.page_source[:500])
print(len(cards))
if len(cards) > 0:
    print(cards[0].text)
