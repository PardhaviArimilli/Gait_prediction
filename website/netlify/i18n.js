// Minimal i18n + RTL toggle (client-side only)
(function(){
  const dict = {
    en: {
      nav_results: 'Results',
      results_title: 'Latest Summaries',
      results_desc: 'Summaries generated from the latest inference runs.',
      banner_title: 'Important:',
      banner_text: 'This tool is for screening only and is not medical advice. If you have urgent symptoms, contact emergency services. Please consult a clinician for diagnosis and care.',
      btn_evaluate: 'Evaluate',
      btn_summary: 'Generate Doctor Summary',
      brand: 'FOG Screening',
      nav_home: 'Home',
      nav_about: 'About',
      title_screening: 'Gait Parameter Screening',
      desc_screening: 'This screening flags potential gait abnormalities via heuristic thresholds. Not a diagnosis.',
      label_cadence: 'Cadence (steps/min)',
      ph_cadence: 'e.g., 110',
      label_stride: 'Stride length (m)',
      ph_stride: 'e.g., 1.2',
      label_stv: 'Step time variability (ms)',
      ph_stv: 'e.g., 35',
      label_symm: 'Symmetry (% diff)',
      ph_symm: 'e.g., 6',
      label_turn: 'Turn duration (s)',
      ph_turn: 'e.g., 2.4',
      label_age: 'Age group',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'Care Finder',
      care_desc: 'Screening flagged abnormal. This is not a diagnosis. Consider consulting a clinician.',
      label_loc: 'Your location (city, address)',
      ph_loc: 'e.g., Bengaluru, India',
      label_radius: 'Radius (km)',
      btn_care: 'Search nearby care',
      about_title: "About Gait and Parkinson's Disease",
      about_p1: "Freezing of gait (FOG) is a brief, episodic absence or marked reduction of forward progression of the feet despite the intention to walk. It is common in Parkinson's disease (PD), particularly in advanced stages, and is linked with falls and reduced quality of life.",
      about_li1: 'Common triggers: turning, doorways, dual-tasking, start hesitation.',
      about_li2: 'Key observable features: increased step time variability, reduced stride length, asymmetry, and prolonged turns.',
      about_li3: 'This tool provides a heuristic screening based on simple thresholds and is not a diagnostic device.',
      about_p2: 'For the ML system in this project, wearable accelerometer data (vertical, medio-lateral, anterior–posterior axes) are used to detect FOG-related events with deep learning models. Evaluation uses cross-validation and ensembling.',
    },
    ar: {
      nav_results: 'النتائج',
      results_title: 'الملخصات الأخيرة',
      results_desc: 'ملخصات تم إنشاؤها من أحدث عمليات الاستدلال.',
      banner_title: 'هام:',
      banner_text: 'هذه الأداة للفحص فقط وليست نصيحة طبية. إذا كانت لديك أعراض طارئة فاتصل بخدمات الطوارئ. يُرجى استشارة الطبيب للتشخيص والرعاية.',
      btn_evaluate: 'تقييم',
      btn_summary: 'إنشاء ملخص للطبيب',
      brand: 'فحص التجمّد',
      nav_home: 'الرئيسية',
      nav_about: 'حول',
      title_screening: 'فحص معلمات المشي',
      desc_screening: 'هذه أداة فحص مؤشرات عدم انتظام المشي وفق قواعد بسيطة. ليست تشخيصًا.',
      label_cadence: 'الإيقاع (خطوات/دقيقة)',
      ph_cadence: 'مثال: 110',
      label_stride: 'طول الخطوة (متر)',
      ph_stride: 'مثال: 1.2',
      label_stv: 'تباين زمن الخطوة (مللي ثانية)',
      ph_stv: 'مثال: 35',
      label_symm: 'التناظر (% فرق)',
      ph_symm: 'مثال: 6',
      label_turn: 'مدة الدوران (ثانية)',
      ph_turn: 'مثال: 2.4',
      label_age: 'الفئة العمرية',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'البحث عن رعاية',
      care_desc: 'النتيجة غير طبيعية. هذه ليست تشخيصًا. يُرجى استشارة الطبيب.',
      label_loc: 'موقعك (مدينة، عنوان)',
      ph_loc: 'مثال: Bengaluru, India',
      label_radius: 'نطاق (كم)',
      btn_care: 'ابحث عن الرعاية القريبة',
      about_title: 'حول المشي ومرض باركنسون',
      about_p1: 'تجمّد المشي هو توقف قصير عن التقدّم رغم نية المشي. شائع في باركنسون ومقترن بالسقوط وتدنّي جودة الحياة.',
      about_li1: 'محرّكات شائعة: الدوران، الأبواب، المهام المزدوجة، التردد عند البدء.',
      about_li2: 'مؤشرات: زيادة تباين زمن الخطوة، قصر الخطوة، عدم التناظر، إطالة الدوران.',
      about_li3: 'هذه أداة فحص وليست جهاز تشخيص.',
      about_p2: 'نستخدم معطيات مسرّع الحركة لاكتشاف أحداث FOG بنماذج تعلم عميق مع تحقق متقاطع وتجميع.',
    },
    hi: {
      banner_title: 'महत्वपूर्ण:',
      banner_text: 'यह उपकरण केवल स्क्रीनिंग के लिए है और यह चिकित्सीय सलाह नहीं है। यदि आपको आपातकालीन लक्षण हैं, तो आपातकालीन सेवाओं से संपर्क करें। कृपया निदान और देखभाल के लिए चिकित्सक से परामर्श करें।',
      btn_evaluate: 'मूल्यांकन',
      btn_summary: 'डॉक्टर सारांश बनाएं',
      brand: 'FOG स्क्रीनिंग',
      nav_home: 'होम',
      nav_about: 'बारे में',
      title_screening: 'गति पैरामीटर स्क्रीनिंग',
      desc_screening: 'यह स्क्रीनिंग सरल नियमों के आधार पर संभावित असामान्यताओं को दर्शाती है। यह निदान नहीं है।',
      label_cadence: 'कैडेंस (कदम/मिनट)',
      ph_cadence: 'जैसे, 110',
      label_stride: 'कदम की लंबाई (मीटर)',
      ph_stride: 'जैसे, 1.2',
      label_stv: 'स्टेप समय परिवर्तनशीलता (मि.से.)',
      ph_stv: 'जैसे, 35',
      label_symm: 'सममिति (% भिन्नता)',
      ph_symm: 'जैसे, 6',
      label_turn: 'मोड़ अवधि (से.)',
      ph_turn: 'जैसे, 2.4',
      label_age: 'आयु समूह',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'देखभाल खोजक',
      care_desc: 'स्क्रीनिंग असामान्य दर्शाती है। यह निदान नहीं है। कृपया चिकित्सक से मिलें।',
      label_loc: 'आपका स्थान (शहर, पता)',
      ph_loc: 'जैसे, बेंगलुरु, भारत',
      label_radius: 'त्रिज्या (कि.मी.)',
      btn_care: 'नज़दीकी देखभाल खोजें',
      about_title: 'गति और पार्किंसंस के बारे में',
      about_p1: 'फ्रीज़िंग ऑफ़ गेट (FOG) चलने की इच्छा होने पर भी पैरों की प्रगति का संक्षिप्त रुक जाना है। यह पार्किंसंस रोग में आम है और गिरावट तथा जीवन-गुणवत्ता में कमी से जुड़ा है।',
      about_li1: 'सामान्य ट्रिगर्स: मुड़ना, दरवाजे, दोहरी-कार्य, शुरू में हिचकिचाहट।',
      about_li2: 'संकेतक: स्टेप समय परिवर्तनशीलता, कम स्ट्राइड लंबाई, असममिति, लंबा टर्न।',
      about_li3: 'यह उपकरण सरल थ्रेशहोल्ड पर आधारित एक स्क्रीनिंग है; यह निदान नहीं है।',
      about_p2: 'इस प्रोजेक्ट में पहने जाने वाले एक्सेलेरोमीटर डाटा से डीप लर्निंग द्वारा FOG घटनाओं का पता लगाया जाता है और क्रॉस-वैलिडेशन/एन्सेम्बलिंग से मूल्यांकन होता है।',
    },
    te: {
      banner_title: 'ముఖ్యం:',
      banner_text: 'ఈ సాధనం స్క్రీనింగ్ కోసం మాత్రమే, ఇది వైద్య సూచన కాదు. అత్యవసర లక్షణాలు ఉంటే ఎమర్జెన్సీ సేవలకు కాల్ చేయండి. దయచేసి నిర్ధారణ మరియు సంరక్షణ కోసం వైద్యుడిని సంప్రదించండి.',
      btn_evaluate: 'మూల్యాంకనం',
      btn_summary: 'డాక్టర్ సారాంశం సృష్టించండి',
      brand: 'FOG స్క్రీనింగ్',
      nav_home: 'హోమ్',
      nav_about: 'గురించి',
      title_screening: 'నడక పరామితి స్క్రీనింగ్',
      desc_screening: 'సాధారణ నియమాల ఆధారంగా సాధ్యమైన అసాధారణాలను సూచిస్తుంది. ఇది నిర్ధారణ కాదు.',
      label_cadence: 'కేడెన్స్ (అడుగులు/ని)',
      ph_cadence: 'ఉదా., 110',
      label_stride: 'స్ట్రైడ్ పొడవు (మీ.)',
      ph_stride: 'ఉదా., 1.2',
      label_stv: 'స్టెప్ టైమ్ వెరి. (మి.సె)',
      ph_stv: 'ఉదా., 35',
      label_symm: 'సమమితం (% తేడా)',
      ph_symm: 'ఉదా., 6',
      label_turn: 'తిరుగు వ్యవధి (సె.)',
      ph_turn: 'ఉదా., 2.4',
      label_age: 'వయసు సమూహం',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'కేర్ ఫైండర్',
      care_desc: 'స్క్రీనింగ్ అసాధారణం చూపింది. ఇది నిర్ధారణ కాదు. దయచేసి వైద్యుడిని సంప్రదించండి.',
      label_loc: 'మీ స్థానం (నగరం, చిరునామా)',
      ph_loc: 'ఉదా., బెంగళూరు, ఇండియా',
      label_radius: 'వ్యాసార్థం (కి.మీ.)',
      btn_care: 'సమీప కేర్ వెతకండి',
      about_title: 'నడక మరియు పార్కిన్సన్ గురించి',
      about_p1: 'FOG అనేది నడవాలనే ఉద్దేశం ఉన్నప్పటికీ కాళ్ల ముందడుగు తాత్కాలికంగా నిలిచిపోవడం. పార్కిన్సన్‌లో ఇది సాధారణం; పతనాలు, జీవన నాణ్యత తగ్గుదలతో సంబంధం ఉంది.',
      about_li1: 'సాధారణ ట్రిగ్గర్లు: తిరుగుడు, తలుపులు, ద్వంద్వ-పని, ప్రారంభ సంకోచం.',
      about_li2: 'సూచకాలు: అధిక స్టెప్ టైమ్ వెరి., తక్కువ స్ట్రైడ్, అసమమితం, ఎక్కువ టర్న్.',
      about_li3: 'ఇది సాధారణ థ్రెషోల్డ్‌లపై ఆధారిత స్క్రీనింగ్ మాత్రమే; నిర్ధారణ కాదు.',
      about_p2: 'ఈ ప్రాజెక్ట్‌లో ధరించే యాక్సిలరోమీటర్ డాటాతో డీప్ లెర్నింగ్ ద్వారా FOG సంఘటనలను గుర్తించి, క్రాస్-వాలిడేషన్/ఎన్సెంబ్లింగ్‌తో మదింపు చేస్తాము.',
    },
    ta: {
      banner_title: 'முக்கியம்:',
      banner_text: 'இந்த கருவி சோதனைக்காக மட்டுமே, இது மருத்துவ ஆலோசனை அல்ல. அவசர அறிகுறிகள் இருந்தால் அவசர சேவைகளை தொடர்பு கொள்ளவும். தயவுசெய்து மருத்துவரை சந்திக்கவும்.',
      btn_evaluate: 'மதிப்பீடு',
      btn_summary: 'மருத்துவர் சுருக்கம் உருவாக்கு',
      brand: 'FOG பரிசோதனை',
      nav_home: 'முகப்பு',
      nav_about: 'பற்றி',
      title_screening: 'நடை அளவுரு பரிசோதனை',
      desc_screening: 'எளிய விதிகளின் அடிப்படையில் சாத்தியமான மாற்றங்களை குறிக்கிறது. இது ஒரு நோயறிதல் அல்ல.',
      label_cadence: 'கேடன்ஸ் (அடிகள்/நிமிடம்)',
      ph_cadence: 'உதா., 110',
      label_stride: 'ஸ்ட்ரைடு நீளம் (மீ.)',
      ph_stride: 'உதா., 1.2',
      label_stv: 'அடிவேளையின் மாறுபாடு (மி.வி.)',
      ph_stv: 'உதா., 35',
      label_symm: 'சமச்சீர் (% வேறுபாடு)',
      ph_symm: 'உதா., 6',
      label_turn: 'முறுக்கு நேரம் (வி.)',
      ph_turn: 'உதா., 2.4',
      label_age: 'வயது குழு',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'சிகிச்சை தேடல்',
      care_desc: 'திரைவு அசாதாரணம் காட்டுகிறது. இது ஒரு நோயறிதல் அல்ல. தயவுசெய்து மருத்துவரை அணுகவும்.',
      label_loc: 'உங்கள் இடம் (நகரம், முகவரி)',
      ph_loc: 'உதா., பெங்களூரு, இந்தியா',
      label_radius: 'விளக்கம் (கிமீ)',
      btn_care: 'அருகிலுள்ள சிகிச்சை தேடு',
      about_title: 'நடை மற்றும் பார்கின்சன் குறித்து',
      about_p1: 'FOG என்பது நடக்க விருப்பமிருந்தும், சற்றுநேரம் கால்களின் முன்னேற்றம் நின்று போவது. பார்கின்சனில் இது பொதுவானது; விழுதல்கள் மற்றும் வாழ்க்கைத் தர குறைவுடன் தொடர்புடையது.',
      about_li1: 'பொதுவான தூண்டுதல்: முறுக்கு, கதவுகள், இரட்டை-பணி, தொடக்க தயக்கம்.',
      about_li2: 'குறியீடுகள்: அதிக அடிவேளை மாறுபாடு, குறைந்த ஸ்ட்ரைடு, அசமச்சீர், நீண்ட முறுக்கு.',
      about_li3: 'இது ஒரு எளிய வரம்புகள் அடிப்படையிலான திரைவு; நோயறிதல் அல்ல.',
      about_p2: 'இந்த திட்டத்தில் அணியக்கூடிய ஆக்சிலரோமீட்டர் தரவிலிருந்து டீப் லெர்னிங் கொண்டு FOG நிகழ்வுகளை கண்டறிந்து, cross-validation மற்றும் ensemble மூலம் மதிப்பிடுகிறோம்.',
    },
    kn: {
      banner_title: 'ಮುಖ್ಯ:',
      banner_text: 'ಈ ಸಾಧನವು ತಪಾಸಣೆಗಾಗಿ ಮಾತ್ರ, ಇದು ವೈದ್ಯಕೀಯ ಸಲಹೆ ಅಲ್ಲ. ತುರ್ತು ಲಕ್ಷಣಗಳಿದ್ದರೆ ತುರ್ತು ಸೇವೆಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ. ದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.',
      btn_evaluate: 'ಮೌಲ್ಯಮಾಪನ',
      btn_summary: 'ವೈದ್ಯರ ಸಾರಾಂಶ ರಚಿತ',
      brand: 'FOG ತಪಾಸಣೆ',
      nav_home: 'ಮುಖಪುಟ',
      nav_about: 'ಬಗ್ಗೆ',
      title_screening: 'ನಡೆ ಪರಿಮಾಣ ತಪಾಸಣೆ',
      desc_screening: 'ಸರಳ ನಿಯಮಗಳ ಆಧಾರದ ಮೇಲೆ ಸಾಧ್ಯ ಅಸಾಮಾನ್ಯತೆಗಳನ್ನು ಸೂಚಿಸುತ್ತದೆ. ಇದು ನಿರ್ಣಯವಲ್ಲ.',
      label_cadence: 'ಕೇಡೆನ್ಸ್ (ಹೆಜ್ಜೆ/ನಿ)',
      ph_cadence: 'ಉದಾ., 110',
      label_stride: 'ಸ್ಟ್ರೈಡ್ ಉದ್ದ (ಮೀ.)',
      ph_stride: 'ಉದಾ., 1.2',
      label_stv: 'ಹೆಜ್ಜೆ ಕಾಲ ವ್ಯತ್ಯಾಸ (ಮಿ.ಸೆ.)',
      ph_stv: 'ಉದಾ., 35',
      label_symm: 'ಸಮಮಿತತೆ (% ವ್ಯತ್ಯಾಸ)',
      ph_symm: 'ಉದಾ., 6',
      label_turn: 'ತಿರುಗುವ ಅವಧಿ (ಸೆ.)',
      ph_turn: 'ಉದಾ., 2.4',
      label_age: 'ವಯೋವರ್ಗ',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'ಕೇರ್ ಫೈಂಡರ್',
      care_desc: 'ತಪಾಸಣೆ ಅಸಾಮಾನ್ಯತೆಯನ್ನು ತೋರಿಸಿದೆ. ಇದು ನಿರ್ಣಯವಲ್ಲ. ದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.',
      label_loc: 'ನಿಮ್ಮ ಸ್ಥಳ (ನಗರ, ವಿಳಾಸ)',
      ph_loc: 'ಉದಾ., ಬೆಂಗಳೂರು, ಭಾರತ',
      label_radius: 'ವ್ಯಾಸಾರ್ಧ (ಕಿ.ಮೀ.)',
      btn_care: 'ಹತ್ತಿರದ ಕೇರ್ ಹುಡುಕಿ',
      about_title: 'ನಡೆ ಮತ್ತು ಪಾರ್ಕಿನ್ಸನ್ಸ್ ಕುರಿತು',
      about_p1: 'FOG ಎಂದರೆ ನಡೆಯುವ ಇಚ್ಛೆ ಇದ್ದೂ ಕಾಲುಗಳ ಮುಂದುವರಿಕೆ ತಾತ್ಕಾಲಿಕವಾಗಿ ನಿಲ್ಲುವುದು. ಪಾರ್ಕಿನ್ಸನ್ಸ್‌ನಲ್ಲಿ ಸಾಮಾನ್ಯ; ಬೀಳಿಕೆ ಮತ್ತು ಜೀವನ ಗುಣಮಟ್ಟ ಕುಂಠಿತವಾಗುತ್ತದೆ.',
      about_li1: 'ಸಾಮಾನ್ಯ ಟ್ರಿಗರ್‌ಗಳು: ತಿರುಗುವುದು, ಬಾಗಿಲುಗಳು, ದ್ವಂದ್ವ-ಕಾರ್ಯ, ಆರಂಭ ಹಿಂಜರಿಕೆ.',
      about_li2: 'ಸೂಚಕಗಳು: ಹೆಚ್ಚಿದ ಹೆಜ್ಜೆ ಕಾಲ ವ್ಯತ್ಯಾಸ, ಕಡಿಮೆ ಸ್ಟ್ರೈಡ್, ಅಸಮಮಿತತೆ, ದೀರ್ಘ ತಿರುಗು.',
      about_li3: 'ಇದು ಸರಳ ಮಿತಿಗಳ ಆಧಾರಿತ ತಪಾಸಣೆ; ನಿರ್ಣಯವಲ್ಲ.',
      about_p2: 'ಈ ಯೋಜನೆಯಲ್ಲಿ ಧರಿಸಬಹುದಾದ ಆಕ್ಸಿಲರೋಮೀಟರ್ ಡೇಟಾದಿಂದ ಡೀಪ್ ಲರ್ನಿಂಗ್ ಮೂಲಕ FOG ಘಟನೆಗಳನ್ನು ಪತ್ತೆ ಮಾಡಿ, ಕ್ರಾಸ್-ವ್ಯಾಲಿಡೇಶನ್/ಎನ್ಸೆಂಬ್ಲಿಂಗ್ ಮೂಲಕ ಮೌಲ್ಯಮಾಪನ ಮಾಡಲಾಗಿದೆ.',
    },
    ml: {
      banner_title: 'പ്രധാന്യം:',
      banner_text: 'ഈ ഉപകരണം സ്ക്രീനിംഗിനായി മാത്രമാണ്, ഇത് മെഡിക്കൽ ഉപദേശം അല്ല. അടിയന്തര ലക്ഷണങ്ങൾ ഉണ്ടെങ്കിൽ അടിയന്തര സേവനങ്ങളെ ബന്ധപ്പെടുക. ദയവായി ഡോക്ടറെ കാണുക.',
      btn_evaluate: 'മൂല്യനിർണ്ണയം',
      btn_summary: 'ഡോക്ടർ സംഗ്രഹം സൃഷ്ടിക്കുക',
      brand: 'FOG സ്ക്രീനിംഗ്',
      nav_home: 'ഹോം',
      nav_about: 'കുറിച്ച്',
      title_screening: 'നടപ്പ് പാരാമീറ്റർ സ്ക്രീനിംഗ്',
      desc_screening: 'ലളിതമായ നിയമങ്ങളെ അടിസ്ഥാനമാക്കി സാധ്യമായ അസാധാരണതകൾ സൂചിപ്പിക്കുന്നു. ഇത് നിരീക്ഷണമാത്രം, രോഗനിർണയം അല്ല.',
      label_cadence: 'കേഡൻസ് (അടി/മിനിറ്റ്)',
      ph_cadence: 'ഉദാ., 110',
      label_stride: 'സ്ട്രൈഡ് നീളം (മീ.)',
      ph_stride: 'ഉദാ., 1.2',
      label_stv: 'സ്റ്റെപ്പ് ടൈം വ്യതിയാനം (മി.സെ.)',
      ph_stv: 'ഉദാ., 35',
      label_symm: 'സിമ്മെട്രി (% വ്യത്യാസം)',
      ph_symm: 'ഉദാ., 6',
      label_turn: 'ടേൺ ദൈർഘ്യം (സെ.)',
      ph_turn: 'ഉദാ., 2.4',
      label_age: 'പ്രായ ഗ്രൂപ്പ്',
      age_adult: '18–64',
      age_older: '65+',
      care_title: 'കെയർ ഫൈൻഡർ',
      care_desc: 'സ്ക്രീനിംഗ് അസാധാരണത കാണിക്കുന്നു. ഇത് രോഗനിർണയം അല്ല. ദയവായി ഡോക്ടറെ കാണുക.',
      label_loc: 'നിങ്ങളുടെ സ്ഥലം (നഗരം, വിലാസം)',
      ph_loc: 'ഉദാ., ബെംഗളൂരു, ഇന്ത്യ',
      label_radius: 'റേഡിയസ് (കിമീ)',
      btn_care: 'അടുത്തുള്ള കെയർ കണ്ടെത്തുക',
      about_title: 'നടപ്പ്‌യും പാർകിൻസൺസും',
      about_p1: 'FOG എന്നാണ് നടപ്പാൻ ഉദ്ദേശമുണ്ടായിട്ടും പാദങ്ങളുടെ മുന്നേറ്റം താൽക്കാലികമായി നിൽക്കുന്നത്. പാർകിൻസണിൽ സാധാരണമാണ്; വീഴ്ചയും ജീവിതഗുണനിലവാരത്തിൽ കുറവും.',
      about_li1: 'സാധാരണ ട്രിഗറുകൾ: തിരിവ്, വാതിലുകൾ, ഇരട്ട-ടാസ്കിംഗ്, തുടക്ക മടിപ്പ്.',
      about_li2: 'സൂചകങ്ങൾ: കൂടിയ സ്റ്റെപ്പ് ടൈം വ്യതിയാനം, കുറഞ്ഞ സ്ട്രൈഡ്, അസിമ്മെട്രി, ദീർഘമായ തിരിവ്.',
      about_li3: 'ഇത് ലളിതമായ പരിധികളിലുള്ള സ്ക്രീനിംഗ് ആണ്; രോഗനിർണയം അല്ല.',
      about_p2: 'ധരിക്കാവുന്ന ആക്സിലറോമീറ്റർ ഡാറ്റ ഉപയോഗിച്ച് ഡീപ്പ് ലേണിംഗ് വഴി FOG സംഭവങ്ങൾ കണ്ടെത്തി ക്രോസ്-വാലിഡേഷൻ/എൻസംബർ ചെയ്യുന്നു.',
    },
  };

  function t(key){
    const lang = window.__i18n_lang || 'en';
    const d = dict[lang] || dict.en;
    return d[key] || (dict.en[key] || key);
  }

  function applyI18n(lang){
    const d = dict[lang] || dict.en;
    window.__i18n_lang = lang;
    document.documentElement.lang = lang;
    document.documentElement.dir = (lang === 'ar') ? 'rtl' : 'ltr';
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (d[key]) el.textContent = d[key];
    });
    document.querySelectorAll('[data-i18n-ph]').forEach(el => {
      const key = el.getAttribute('data-i18n-ph');
      if (d[key]) el.setAttribute('placeholder', d[key]);
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      const key = el.getAttribute('data-i18n-title');
      if (d[key]) el.setAttribute('title', d[key]);
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    // Dropdown hover behavior
    const menu = document.querySelector('.lang-menu');
    const trigger = document.querySelector('.lang-trigger');
    const dropdown = document.querySelector('.lang-dropdown');
    if (menu && trigger && dropdown) {
      let hover = false; let t;
      const show = () => { dropdown.style.display = 'block'; };
      const hide = () => { dropdown.style.display = 'none'; };
      menu.addEventListener('mouseenter', () => { hover = true; clearTimeout(t); show(); });
      menu.addEventListener('mouseleave', () => { hover = false; t = setTimeout(hide, 150); });
      dropdown.querySelectorAll('.lang-option').forEach(btn => {
        btn.addEventListener('click', () => {
          const lang = btn.getAttribute('data-lang');
          applyI18n(lang);
          hide();
        });
      });
    }
    applyI18n('en');
    window.i18n = { t, applyI18n, dict };
  });
})();


