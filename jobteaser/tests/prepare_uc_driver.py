import undetected_chromedriver as uc

if __name__ == "__main__":
    print("🔧 Initialisation undetected_chromedriver (one-time)")
    driver = uc.Chrome(headless=True)
    driver.quit()
    print("✔ Driver prêt")
