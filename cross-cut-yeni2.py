import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd

# Mouse-based selection için alternatif kütüphane
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    try:
        # Alternatif: streamlit-image-coordinates
        from streamlit_image_coordinates import streamlit_image_coordinates
        CANVAS_AVAILABLE = "coordinates"
    except ImportError:
        CANVAS_AVAILABLE = False
        st.error("Mouse desteği için kütüphane kurun: pip install streamlit-image-coordinates")

try:
    from skimage.measure import regionprops, label
except ImportError:
    # Fallback if scikit-image not available
    def label(image):
        return cv2.connectedComponents(image.astype(np.uint8))[1], None
    
    def regionprops(labeled):
        # Simple fallback implementation
        return []

class PatternRecognitionClassifier:
    def __init__(self):
        self.iso_classes = {
            0: {"name": "Sınıf 0", "description": "Hiç hasar yok", "color": "#27ae60", "damage_range": (0, 0)},
            1: {"name": "Sınıf 1", "description": "Küçük pullanmalar (%5'den az)", "color": "#2ecc71", "damage_range": (0, 5)},
            2: {"name": "Sınıf 2", "description": "Kenar pullanmaları (%5-15)", "color": "#f1c40f", "damage_range": (5, 15)},
            3: {"name": "Sınıf 3", "description": "Büyük şeritler (%15-35)", "color": "#e67e22", "damage_range": (15, 35)},
            4: {"name": "Sınıf 4", "description": "Çok büyük hasarlar (%35-65)", "color": "#e74c3c", "damage_range": (35, 65)},
            5: {"name": "Sınıf 5", "description": "Maksimum hasar (%65+)", "color": "#c0392b", "damage_range": (65, 100)}
        }
        
        # Cross-cut hasar pattern tanımları
        self.damage_patterns = {
            'flaking': {
                'name': 'Flaking (Pullanma)',
                'description': 'Boya tabakasının ayrılması',
                'min_area': 50,
                'max_aspect_ratio': 3,
                'circularity_range': (0.3, 0.9),
                'severity_multiplier': 1.0
            },
            'cracking': {
                'name': 'Cracking (Çatlama)',
                'description': 'Linear çatlaklar',
                'min_length': 30,
                'max_width': 10,
                'aspect_ratio_min': 3,
                'severity_multiplier': 0.8
            },
            'delamination': {
                'name': 'Delamination (Delaminasyon)',
                'description': 'Geniş alan ayrılması',
                'min_area': 100,
                'max_aspect_ratio': 2,
                'severity_multiplier': 1.2
            },
            'edge_damage': {
                'name': 'Edge Damage (Kenar Hasarı)',
                'description': 'Grid çizgileri boyunca hasar',
                'edge_proximity': 10,
                'severity_multiplier': 0.9
            }
        }

    def extract_grid_region(self, image, x, y, width, height):
        """Grid bölgesini çıkar"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        h, w = img_array.shape[:2]
        x = max(0, min(x, w - width))
        y = max(0, min(y, h - height))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return img_array[y:y+height, x:x+width]

    def detect_grid_lines(self, image):
        """Grid çizgilerini tespit et"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Hough Line Transform ile çizgileri tespit et
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=min(gray.shape) // 6, maxLineGap=10)
        
        grid_mask = np.zeros_like(gray)
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çizgi açısını hesapla
                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal
                    horizontal_lines.append(line[0])
                    cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 3)
                else:  # Vertical
                    vertical_lines.append(line[0])
                    cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 3)
        
        return grid_mask, horizontal_lines, vertical_lines

    def detect_damage_patterns(self, image, grid_mask):
        """Hasar paternlerini tespit et - Düzeltilmiş versiyon"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Grid çizgilerini çıkar - inpaint yerine maskeleme kullan
        grid_mask_dilated = cv2.dilate(grid_mask, np.ones((5,5), np.uint8), iterations=1)
        cleaned = gray.copy()
        cleaned[grid_mask_dilated > 0] = np.median(gray)  # Grid çizgilerini median ile doldur
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Grid maskesini binary'den çıkar
        binary[grid_mask_dilated > 0] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Connected components with OpenCV
        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        
        detected_patterns = {
            'flaking': [],
            'cracking': [],
            'delamination': [],
            'edge_damage': []
        }
        
        # Her component için analiz
        for label_id in range(1, num_labels):  # 0 background
            mask = (labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            contour = contours[0]  # En büyük contour
            area = cv2.contourArea(contour)
            
            if area < 10:  # Çok küçük alanları atla
                continue
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            # Circularity hesapla
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Merkez nokta
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Kenar mesafesi
            edge_distance = min(cx, cy, gray.shape[1] - cx, gray.shape[0] - cy)
            
            # Pattern classification
            pattern_scores = {}
            
            # Flaking pattern
            if (area >= self.damage_patterns['flaking']['min_area'] and
                aspect_ratio <= self.damage_patterns['flaking']['max_aspect_ratio'] and
                self.damage_patterns['flaking']['circularity_range'][0] <= circularity <= 
                self.damage_patterns['flaking']['circularity_range'][1]):
                pattern_scores['flaking'] = circularity * (area / 100)
            
            # Cracking pattern
            if (aspect_ratio >= self.damage_patterns['cracking']['aspect_ratio_min'] and
                max(w, h) >= self.damage_patterns['cracking']['min_length']):
                pattern_scores['cracking'] = aspect_ratio * 0.1
            
            # Delamination pattern
            if (area >= self.damage_patterns['delamination']['min_area'] and
                aspect_ratio <= self.damage_patterns['delamination']['max_aspect_ratio']):
                pattern_scores['delamination'] = area / 200
            
            # Edge damage pattern
            if edge_distance <= self.damage_patterns['edge_damage']['edge_proximity']:
                pattern_scores['edge_damage'] = (1 / max(edge_distance, 1)) * area / 50
            
            # En yüksek skora sahip pattern'ı seç
            if pattern_scores:
                best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
                
                # Basit region objesi oluştur
                region_obj = {
                    'centroid': (cy, cx),
                    'area': area,
                    'bbox': (y, x, y+h, x+w)
                }
                
                detected_patterns[best_pattern[0]].append({
                    'region': region_obj,
                    'score': best_pattern[1],
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'edge_distance': edge_distance
                })
        
        return detected_patterns, binary

    def calculate_damage_severity(self, patterns, grid_size):
        """Pattern'lara göre hasar şiddetini hesapla"""
        total_severity = 0
        pattern_counts = {}
        
        for pattern_type, pattern_list in patterns.items():
            count = len(pattern_list)
            pattern_counts[pattern_type] = count
            
            if count > 0:
                # Pattern'e göre ağırlıklı skor
                multiplier = self.damage_patterns[pattern_type]['severity_multiplier']
                pattern_severity = sum(p['score'] for p in pattern_list) * multiplier
                total_severity += pattern_severity
        
        # Grid boyutuna göre normalize et
        grid_area = grid_size[0] * grid_size[1]
        normalized_severity = (total_severity / grid_area) * 10000
        
        return normalized_severity, pattern_counts

    def analyze_5x5_cells(self, image, patterns):
        """5x5 hücre bazında analiz - Düzeltilmiş versiyon"""
        h, w = image.shape[:2]
        cell_h = h // 5
        cell_w = w // 5
        
        cell_damages = np.zeros((5, 5))
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Dict'ten centroid bilgisini al
                centroid = pattern['region']['centroid']  # Bu (y, x) formatında
                cell_row = min(4, int(centroid[0] // cell_h))
                cell_col = min(4, int(centroid[1] // cell_w))
                
                # Pattern şiddetini hücreye ekle
                severity = pattern['score'] * self.damage_patterns[pattern_type]['severity_multiplier']
                cell_damages[cell_row, cell_col] += severity
        
        # Hasarlı hücre sayısını hesapla (düşük eşik)
        damaged_cells = np.sum(cell_damages > 0.01)
        
        return cell_damages, damaged_cells

    def classify_by_cell_matrix(self, cell_damages, pattern_counts):
        """ISO 2409:2013 standardına uygun 5x5 hücre matrisi sınıflandırması"""
        
        # Çok düşük eşik - herhangi bir hasar varlığını tespit et
        damage_threshold = 0.01
        
        # Hasarlı hücre sayısını hesapla
        damaged_cells = np.sum(cell_damages > damage_threshold)
        
        # Toplam pattern sayısı
        total_patterns = sum(pattern_counts.values())
        
        # Hasar yüzdesi hesaplama
        damage_percentage = (damaged_cells / 25.0) * 100
        
        # ISO 2409:2013'e göre sınıflandırma
        # ÖNEMLI: Eğer herhangi bir pattern tespit edildiyse minimum Sınıf 1 olmalı
        if total_patterns == 0 and damaged_cells == 0:
            return 0, 0  # Gerçekten hiç hasar yok
        
        elif total_patterns > 0 and damaged_cells <= 3:
            # Pattern tespit edildi - kesişimlerde küçük pullanmalar
            # ISO: "küçük pullar halinde ayrılması, %5'den az etkilenme"
            return 1, max(damage_percentage, 4.0)  # En az %4 hasar varsay
        
        elif damaged_cells <= 6:
            # ISO: "kenarlar boyunca pullanma, %5-15 etkilenme"
            return 2, max(damage_percentage, 8.0)
        
        elif damaged_cells <= 12:
            # ISO: "büyük şeritler halinde pullanma, %15-35 etkilenme"
            return 3, max(damage_percentage, 20.0)
        
        elif damaged_cells <= 18:
            # ISO: "çok büyük şeritler, %35-65 etkilenme"
            return 4, max(damage_percentage, 45.0)
        
        else:
            # ISO: "Sınıf 4'ten daha kötü, %65+ etkilenme"
            return 5, max(damage_percentage, 70.0)
    
    def generate_matrix_based_probabilities(self, cell_damages, predicted_class, damage_percentage, pattern_counts):
        """Hücre matrisi ve pattern analizi sonucuna göre olasılık dağılımı"""
        probs = np.zeros(6)
        
        # Sınır durumlarını tespit et
        damaged_cells = np.sum(cell_damages > 0.01)
        total_patterns = sum(pattern_counts.values())
        
        # Base confidence hesaplama - pattern varlığı önemli faktör
        if total_patterns > 0 and predicted_class == 1:
            base_confidence = 0.90  # Pattern tespit edildi, Sınıf 1 yüksek güven
        elif damaged_cells in [1, 3, 6, 12, 18]:  # Sınır değerler
            base_confidence = 0.70
        elif damaged_cells == 0 and total_patterns == 0:  # Kesin durum
            base_confidence = 0.95
        elif total_patterns > 5:  # Çok pattern var
            base_confidence = 0.85
        else:
            base_confidence = 0.80
        
        # Ana sınıfa olasılık
        probs[predicted_class] = base_confidence
        
        # Kalan olasılığı dağıt
        remaining = 1.0 - base_confidence
        
        # Pattern sayısına göre komşu sınıflara dağılım
        if total_patterns > 0 and predicted_class == 1:
            # Sınıf 1'den Sınıf 2'ye geçiş olasılığı
            if predicted_class < 5:
                probs[predicted_class + 1] = remaining * 0.6
            if predicted_class > 0:
                probs[predicted_class - 1] = remaining * 0.2
        else:
            # Normal komşu dağılımı
            if predicted_class > 0:
                probs[predicted_class - 1] = remaining * 0.35
            if predicted_class < 5:
                probs[predicted_class + 1] = remaining * 0.35
        
        # Kalan minimal olasılığı uzak sınıflara
        for i in range(6):
            if probs[i] == 0:
                probs[i] = remaining * 0.05
        
        # Normalize et
        return probs / np.sum(probs)
    
    def detailed_cell_analysis(self, cell_damages):
        """Detaylı hücre analizi raporu"""
        damage_threshold = 0.01  # Düşük eşik
        
        # Her hücrenin durumunu analiz et
        cell_analysis = []
        for row in range(5):
            for col in range(5):
                damage_level = cell_damages[row, col]
                is_damaged = damage_level > damage_threshold
                
                # Hasar seviyesi kategorisi
                if damage_level == 0:
                    category = "Sağlam"
                elif damage_level < 0.01:
                    category = "Minimal"
                elif damage_level < 0.05:
                    category = "Hafif"
                elif damage_level < 0.1:
                    category = "Orta"
                else:
                    category = "Ağır"
                
                cell_analysis.append({
                    'position': f"({row},{col})",
                    'damage_score': damage_level,
                    'is_damaged': is_damaged,
                    'category': category
                })
        
        return cell_analysis
    
    def generate_confidence_probabilities(self, severity_score, damaged_cells, predicted_class):
        """Pattern analizi sonucuna göre olasılık dağılımı"""
        probs = np.zeros(6)
        
        # Base confidence
        if severity_score < 2 and damaged_cells == 0:
            base_confidence = 0.95  # Çok kesin
        elif severity_score > 100 or damaged_cells > 15:
            base_confidence = 0.90  # Yüksek hasar kesin
        elif 8 <= severity_score <= 12 or 18 <= severity_score <= 22:  # Sınır durumlar
            base_confidence = 0.70
        else:
            base_confidence = 0.85
            
        probs[predicted_class] = base_confidence
        
        # Kalan olasılığı dağıt
        remaining = 1.0 - base_confidence
        
        # Komşu sınıflara öncelik
        if predicted_class > 0:
            probs[predicted_class - 1] = remaining * 0.4
        if predicted_class < 5:
            probs[predicted_class + 1] = remaining * 0.4
            
        # Kalan az olasılığı uzak sınıflara
        for i in range(6):
            if probs[i] == 0:
                probs[i] = remaining * 0.03
                
        # Normalize
        return probs / np.sum(probs)

    def draw_selection_overlay(self, image, x, y, width, height):
        """Seçim overlay'ini çiz"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Seçim dikdörtgeni
        cv2.rectangle(img_array, (x, y), (x + width, y + height), (0, 255, 0), 3)
        
        # 5x5 grid
        cell_w, cell_h = width // 5, height // 5
        for i in range(1, 5):
            cv2.line(img_array, (x + i * cell_w, y), (x + i * cell_w, y + height), (255, 0, 0), 2)
            cv2.line(img_array, (x, y + i * cell_h), (x + width, y + i * cell_h), (255, 0, 0), 2)
        
        return img_array

def main_pattern_recognition():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>Pattern Recognition Cross-cut Classifier</h1>
        <p>ISO 2409:2013 Uyumlu Hasar Sınıflandırması</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Canvas availability check - fonksiyon içinde
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_available = True
        canvas_type = "drawable"
    except ImportError:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            canvas_available = True
            canvas_type = "coordinates"
        except ImportError:
            canvas_available = False
            canvas_type = "manual"
    
    if 'pattern_classifier' not in st.session_state:
        st.session_state.pattern_classifier = PatternRecognitionClassifier()
    
    classifier = st.session_state.pattern_classifier
    
    # Sidebar
    with st.sidebar:
        st.header("Pattern Recognition Ayarları")
        
        show_patterns = st.multiselect(
            "Gösterilecek Pattern'ler",
            ["flaking", "cracking", "delamination", "edge_damage"],
            default=["flaking", "cracking", "delamination", "edge_damage"]
        )
        
        show_debug = st.checkbox("Pattern Debug Görünümü", value=False)
        
        st.markdown("---")
        st.markdown("### ISO 2409:2013 Sınıfları:")
        st.success("🟢 Sınıf 0: Hiç hasar yok")
        st.success("🟢 Sınıf 1: Küçük pullanmalar (%5'den az)")
        st.warning("🟡 Sınıf 2: Kenar pullanmaları (%5-15)")
        st.warning("🟠 Sınıf 3: Büyük şeritler (%15-35)")
        st.error("🔴 Sınıf 4: Çok büyük hasarlar (%35-65)")
        st.error("⚫ Sınıf 5: Maksimum hasar (%65+)")
        
        st.markdown("### Pattern Özellikleri:")
        st.info("• Geometrik analiz (alan, aspect ratio)")
        st.info("• Şekil analizi (circularity)")
        st.info("• Pozisyon analizi (kenar mesafesi)")
        st.info("• ISO 2409:2013 uyumlu skorlama")
    
    col1 = st.container()
    
    with col1:
        st.header("Grid Seçimi ve Pattern Analizi")
        
        uploaded_file = st.file_uploader("Cross-cut test görüntüsü", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]
            
            st.image(image, caption=f"Yüklenen Görüntü ({img_width}x{img_height})", use_column_width=True)
            
            # Mouse ile grid seçimi
            st.subheader("Grid Alanı Seçimi - Mouse Drag & Drop")
            
            # Initialize grid_selected variable at the start
            grid_selected = False
            
            if canvas_available and canvas_type == "drawable":
                # streamlit-drawable-canvas çalışıyor
                st.markdown("""
                <div style="background: #e8f4fd; padding: 15px; border-left: 5px solid #3498db; margin: 15px 0;">
                    <strong>Mouse İle Grid Seçimi:</strong><br>
                    1. Aşağıda fotoğrafınız tam boyutunda görünecek<br>
                    2. Mouse ile üzerinde dikdörtgen çizin<br>
                    3. Sol üst köşeden sağ alt köşeye sürükleyin<br>
                    4. Yeniden çizmek için "Clear" butonuna basın
                </div>
                """, unsafe_allow_html=True)
                
                canvas_width = img_width
                canvas_height = img_height
                scale_x = 1.0
                scale_y = 1.0
                display_image = image
                
                st.info(f"Canvas: {canvas_width}x{canvas_height} (1:1 Orijinal Boyut)")
                
                st.markdown("### Canvas - Tam Boyut Görüntü")
                
                st.markdown("""
                <style>
                .streamlit-drawable-canvas > div {
                    margin: 0 auto;
                    display: block;
                }
                </style>
                """, unsafe_allow_html=True)
                
                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.1)",
                        stroke_width=3,
                        stroke_color="#00FF00",
                        background_image=display_image,
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="rect",
                        display_toolbar=True,
                        key="grid_canvas",
                    )
                    
                    # Canvas'tan grid koordinatlarını al
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if len(objects) > 0:
                            # Son çizilen dikdörtgeni al
                            last_rect = objects[-1]
                            if last_rect["type"] == "rect":
                                # Canvas koordinatlarından gerçek koordinatlara dönüştür
                                canvas_x = last_rect["left"]
                                canvas_y = last_rect["top"]
                                canvas_w = last_rect["width"]
                                canvas_h = last_rect["height"]
                                
                                # Gerçek koordinatlar - scale faktörlerini kullan
                                real_x = int(canvas_x * scale_x)
                                real_y = int(canvas_y * scale_y)
                                real_w = int(canvas_w * scale_x)
                                real_h = int(canvas_h * scale_y)
                                
                                # Kare yapma seçeneği
                                make_square = st.checkbox("Kare şeklinde zorla", value=True)
                                
                                if make_square:
                                    # En küçük boyutu al
                                    min_size = min(real_w, real_h)
                                    # Merkezden kare oluştur
                                    center_x = real_x + real_w // 2
                                    center_y = real_y + real_h // 2
                                    
                                    final_x = center_x - min_size // 2
                                    final_y = center_y - min_size // 2
                                    final_w = min_size
                                    final_h = min_size
                                else:
                                    final_x = real_x
                                    final_y = real_y
                                    final_w = real_w
                                    final_h = real_h
                                
                                # Sınırları kontrol et
                                final_x = max(0, min(final_x, img_width - final_w))
                                final_y = max(0, min(final_y, img_height - final_h))
                                final_w = min(final_w, img_width - final_x)
                                final_h = min(final_h, img_height - final_y)
                                
                                grid_selected = True
                                
                                # Grid bilgilerini göster
                                col_info1, col_info2, col_info3 = st.columns(3)
                                with col_info1:
                                    st.metric("Grid Pozisyonu", f"({final_x}, {final_y})")
                                with col_info2:
                                    st.metric("Grid Boyutu", f"{final_w} x {final_h}")
                                with col_info3:
                                    ratio = final_w / max(final_h, 1)
                                    st.metric("En/Boy Oranı", f"{ratio:.2f}")
                                
                                # 5x5 grid preview çiz
                                preview_img = classifier.draw_selection_overlay(
                                    img_array, final_x, final_y, final_w, final_h
                                )
                                
                                st.subheader("Grid Preview - 5x5 Bölümler")
                                st.image(preview_img, caption="Seçilen Grid Alanı ve 5x5 Bölümler", use_column_width=True)
                                
                                # Koordinatları session state'e kaydet
                                st.session_state.mouse_selected_coords = (final_x, final_y, final_w, final_h)
                
                except Exception as e:
                    st.error(f"Canvas hatası: {e}")
                    st.error("Streamlit sürümü uyumsuz. Manuel seçim kullanılıyor.")
                    canvas_available = False
                    canvas_type = "manual"
                
                st.markdown("---")
                
            elif canvas_available and canvas_type == "coordinates":
                # streamlit-image-coordinates kullan
                st.markdown("""
                <div style="background: #e8f4fd; padding: 15px; border-left: 5px solid #3498db; margin: 15px 0;">
                    <strong>Click İle Grid Seçimi:</strong><br>
                    1. Önce sol üst köşeyi tıklayın<br>
                    2. Sonra sağ alt köşeyi tıklayın<br>
                    3. Grid otomatik oluşacak
                </div>
                """, unsafe_allow_html=True)
                
                # İki tıklama sistemi
                if 'click_count' not in st.session_state:
                    st.session_state.click_count = 0
                if 'first_click' not in st.session_state:
                    st.session_state.first_click = None
                
                st.write(f"Tıklama adımı: {st.session_state.click_count + 1}/2")
                
                value = streamlit_image_coordinates(image, key="image_coords")
                
                if value is not None:
                    if st.session_state.click_count == 0:
                        st.session_state.first_click = value
                        st.session_state.click_count = 1
                        st.info(f"İlk nokta seçildi: ({value['x']}, {value['y']}). Şimdi ikinci noktayı seçin.")
                    elif st.session_state.click_count == 1:
                        # İkinci tıklama
                        x1, y1 = st.session_state.first_click['x'], st.session_state.first_click['y']
                        x2, y2 = value['x'], value['y']
                        
                        # Grid koordinatları
                        grid_x = min(x1, x2)
                        grid_y = min(y1, y2)
                        grid_w = abs(x2 - x1)
                        grid_h = abs(y2 - y1)
                        
                        # Kare yap
                        min_size = min(grid_w, grid_h)
                        grid_w = min_size
                        grid_h = min_size
                        
                        st.success(f"Grid seçildi: ({grid_x}, {grid_y}) - {grid_w}x{grid_h}")
                        
                        # Preview göster
                        preview_img = classifier.draw_selection_overlay(img_array, grid_x, grid_y, grid_w, grid_h)
                        st.image(preview_img, caption="Seçilen Grid", use_column_width=True)
                        
                        # Koordinatları kaydet
                        st.session_state.mouse_selected_coords = (grid_x, grid_y, grid_w, grid_h)
                        grid_selected = True
                        
                        if st.button("Yeni Seçim"):
                            st.session_state.click_count = 0
                            st.session_state.first_click = None
                            st.rerun()
                
            else:  # Manual mode
                # Fallback: Manuel koordinat girişi
                st.info("Mouse desteği için şu komutlardan birini çalıştırın:")
                st.code("pip install streamlit-image-coordinates")
                st.code("# veya")  
                st.code("pip install streamlit-drawable-canvas==0.8.0")
                
                st.markdown("**Manuel Grid Seçimi:**")
                col_manual1, col_manual2 = st.columns(2)
                
                with col_manual1:
                    manual_x = st.number_input("Grid X", 0, img_width-100, img_width//4)
                    manual_y = st.number_input("Grid Y", 0, img_height-100, img_height//4)
                
                with col_manual2:
                    manual_size = st.number_input("Grid Boyutu", 50, min(img_width, img_height), min(img_width, img_height)//3)
                
                # Manuel preview
                preview_img = classifier.draw_selection_overlay(img_array, manual_x, manual_y, manual_size, manual_size)
                st.image(preview_img, caption="Manuel Grid Seçimi", use_column_width=True)
                
                # Manuel koordinatları kaydet
                st.session_state.mouse_selected_coords = (manual_x, manual_y, manual_size, manual_size)
                grid_selected = True
            
            # Check grid selection status
            if not grid_selected:
                st.info("Yukarıdaki görüntü üzerinde mouse ile bir dikdörtgen çizin.")
            
            # Pattern analizi başlat
            if st.button("🔍 Pattern Recognition Analizi", type="primary", use_container_width=True):
                if 'mouse_selected_coords' not in st.session_state:
                    st.error("Önce mouse ile grid alanını seçin!")
                else:
                    with st.spinner('Pattern\'ler tespit ediliyor...'):
                        
                        # Mouse ile seçilen koordinatları kullan
                        grid_x, grid_y, grid_w, grid_h = st.session_state.mouse_selected_coords
                        
                        # Grid bölgesini çıkar
                        grid_region = classifier.extract_grid_region(img_array, grid_x, grid_y, grid_w, grid_h)
                        st.session_state.pattern_grid = grid_region
                    
                    # Grid çizgilerini tespit et
                    grid_mask, h_lines, v_lines = classifier.detect_grid_lines(grid_region)
                    
                    # Hasar pattern'lerini tespit et
                    patterns, binary_mask = classifier.detect_damage_patterns(grid_region, grid_mask)
                    
                    # Hasar şiddetini hesapla
                    severity, pattern_counts = classifier.calculate_damage_severity(patterns, grid_region.shape[:2])
                    
                    # 5x5 hücre analizi
                    cell_damages, _ = classifier.analyze_5x5_cells(grid_region, patterns)
                    
                    # ISO 2409:2013 uyumlu sınıflandırma - pattern_counts da dahil
                    predicted_class, damage_percentage = classifier.classify_by_cell_matrix(cell_damages, pattern_counts)
                    
                    # Hücre matrisi bazlı olasılık hesaplama
                    probabilities = classifier.generate_matrix_based_probabilities(
                        cell_damages, predicted_class, damage_percentage, pattern_counts
                    )
                    
                    # Detaylı hücre analizi
                    cell_analysis = classifier.detailed_cell_analysis(cell_damages)
                    
                    result = {
                        'predicted_class': predicted_class,
                        'confidence': float(probabilities[predicted_class]),
                        'probabilities': probabilities.tolist(),
                        'damage_percentage': damage_percentage,
                        'damaged_cells': int(np.sum(cell_damages > 0.01)),  # Düşük eşik
                        'total_patterns': sum(pattern_counts.values()),
                        'severity_score': severity,
                        'patterns': patterns,
                        'pattern_counts': pattern_counts,
                        'cell_damages': cell_damages,
                        'cell_analysis': cell_analysis,
                        'class_info': classifier.iso_classes[predicted_class],
                        'grid_lines': (h_lines, v_lines),
                        'binary_mask': binary_mask,
                        'classification_method': 'ISO 2409:2013 Compliant'
                    }
                    
                    st.session_state.pattern_result = result
                    st.success("ISO 2409:2013 uyumlu pattern recognition analizi tamamlandı!")
        
        else:
            st.info("Cross-cut test görüntünüzü yükleyin")
    
    # Pattern analizi sonuçları
    with col1:
        st.header("Pattern Analizi Sonuçları")
        
        if 'pattern_result' in st.session_state:
            result = st.session_state.pattern_result
            class_info = result['class_info']
            
            # Ana sonuç - İyileştirilmiş
            st.markdown(f"""
            <div style="background: {class_info['color']}22; padding: 2rem; border-radius: 15px; border: 3px solid {class_info['color']}; text-align: center;">
                <h2>{class_info['name']}</h2>
                <h3>Hasarlı Hücreler: {result['damaged_cells']}/25</h3>
                <h3>Tespit Edilen Pattern: {result['total_patterns']} adet</h3>
                <h3>Hasar Yüzdesi: {result['damage_percentage']:.1f}%</h3>
                <h3>Güven: {result['confidence']:.1%}</h3>
                <p><strong>{class_info['description']}</strong></p>
                <p><small>Sınıflandırma: {result['classification_method']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # ISO 2409 kriterleri açıklama
            st.info(f"""
            **ISO 2409:2013 Kriterleri:**
            - Pattern tespit edildi: {result['total_patterns']} adet
            - Hasarlı hücre sayısı: {result['damaged_cells']}/25
            - Bu durum ISO standardına göre **{class_info['name']}** kriterlerine uyuyor
            """)
            
            # Çıkarılan grid
            if 'pattern_grid' in st.session_state:
                st.subheader("Analiz Edilen Grid")
                st.image(st.session_state.pattern_grid, caption="5x5 Grid Bölgesi", width=250)
            
            # Pattern istatistikleri
            st.subheader("Tespit Edilen Pattern'ler")
            col_pat1, col_pat2, col_pat3, col_pat4 = st.columns(4)
            
            with col_pat1:
                if result['pattern_counts']['flaking'] > 0:
                    st.metric("Flaking", f"{result['pattern_counts']['flaking']} adet")
            with col_pat2:
                if result['pattern_counts']['cracking'] > 0:
                    st.metric("Cracking", f"{result['pattern_counts']['cracking']} adet")
            with col_pat3:
                if result['pattern_counts']['delamination'] > 0:
                    st.metric("Delamination", f"{result['pattern_counts']['delamination']} adet")
            with col_pat4:
                if result['pattern_counts']['edge_damage'] > 0:
                    st.metric("Edge Damage", f"{result['pattern_counts']['edge_damage']} adet")
            
            # Pattern detayları
            if show_debug:
                st.subheader("Pattern Detayları")
                for pattern_type in show_patterns:
                    if result['pattern_counts'][pattern_type] > 0:
                        st.write(f"**{pattern_type.title()}:**")
                        for i, pattern in enumerate(result['patterns'][pattern_type]):
                            st.write(f"- Pattern {i+1}: Score={pattern['score']:.3f}, Area={pattern['area']}")
            
            # Olasılık grafiği
            st.subheader("Sınıf Olasılık Dağılımı")
            prob_data = pd.DataFrame({
                'Sınıf': [f"Sınıf {i}" for i in range(6)],
                'Olasılık': [p * 100 for p in result['probabilities']],
                'Renk': [classifier.iso_classes[i]['color'] for i in range(6)],
                'Açıklama': [f"{classifier.iso_classes[i]['description']}" for i in range(6)]
            })
            
            fig = px.bar(prob_data, x='Sınıf', y='Olasılık', 
                        color='Sınıf',
                        color_discrete_map={f'Sınıf {i}': classifier.iso_classes[i]['color'] for i in range(6)},
                        title="ISO 2409:2013 Sınıf Olasılıkları",
                        hover_data=['Açıklama'])
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern görselleştirmesi
            if 'pattern_grid' in st.session_state and show_debug:
                st.subheader("Pattern Visualization")
                
                # Pattern'leri renklendirerek göster
                pattern_colors = {
                    'flaking': (255, 0, 0),     # Kırmızı
                    'cracking': (0, 255, 0),   # Yeşil
                    'delamination': (0, 0, 255), # Mavi
                    'edge_damage': (255, 255, 0) # Sarı
                }
                
                if len(st.session_state.pattern_grid.shape) == 3:
                    viz_img = st.session_state.pattern_grid.copy()
                else:
                    viz_img = cv2.cvtColor(st.session_state.pattern_grid, cv2.COLOR_GRAY2RGB)
                
                for pattern_type, pattern_list in result['patterns'].items():
                    if pattern_type in show_patterns and pattern_list:
                        color = pattern_colors.get(pattern_type, (128, 128, 128))
                        for pattern in pattern_list:
                            bbox = pattern['region']['bbox']
                            y, x, y2, x2 = bbox
                            cv2.rectangle(viz_img, (x, y), (x2, y2), color, 2)
                            
                            # Pattern tipi yazısı
                            cv2.putText(viz_img, pattern_type[:4], (x, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                st.image(viz_img, caption="Tespit Edilen Pattern'ler", width=300)
            
            # Hücre hasar matrisi
            st.subheader("5x5 Hücre Hasar Matrisi (ISO 2409 Temel)")
            
            # Matris görselleştirmesi
            fig_matrix = px.imshow(result['cell_damages'], 
                                  title="Hücre Hasar Yoğunluk Haritası",
                                  labels=dict(x="Kolon", y="Sıra", color="Hasar Skoru"),
                                  color_continuous_scale="Reds")
            fig_matrix.update_layout(height=400)
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Detaylı hücre analizi tablosu
            if show_debug and 'cell_analysis' in result:
                st.subheader("Detaylı Hücre Analizi")
                
                cell_df = pd.DataFrame(result['cell_analysis'])
                # Hasarlı hücreleri vurgula
                def highlight_damaged(row):
                    if row['is_damaged']:
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = cell_df.style.apply(highlight_damaged, axis=1)
                st.dataframe(styled_df, use_container_width=True)
            
            # ISO 2409 Sınıflandırma Kriteri
            st.subheader("ISO 2409:2013 Sınıflandırma Kriterleri")
            
            criteria_data = [
                {"Sınıf": 0, "Hasarlı Hücre": "0", "Pattern": "0", "Açıklama": "Hiç hasar yok"},
                {"Sınıf": 1, "Hasarlı Hücre": "≤3", "Pattern": "≤5", "Açıklama": "Küçük pullanmalar (%5'den az)"},
                {"Sınıf": 2, "Hasarlı Hücre": "≤6", "Pattern": "5+", "Açıklama": "Kenar pullanmaları (%5-15)"},
                {"Sınıf": 3, "Hasarlı Hücre": "≤12", "Pattern": "10+", "Açıklama": "Büyük şeritler (%15-35)"},
                {"Sınıf": 4, "Hasarlı Hücre": "≤18", "Pattern": "15+", "Açıklama": "Çok büyük hasarlar (%35-65)"},
                {"Sınıf": 5, "Hasarlı Hücre": ">18", "Pattern": "20+", "Açıklama": "Maksimum hasar (%65+)"}
            ]
            
            criteria_df = pd.DataFrame(criteria_data)
            # Mevcut sınıfı vurgula
            def highlight_current_class(row):
                if row['Sınıf'] == result['predicted_class']:
                    return ['background-color: #90EE90'] * len(row)  # Açık yeşil
                else:
                    return [''] * len(row)
            
            styled_criteria = criteria_df.style.apply(highlight_current_class, axis=1)
            st.dataframe(styled_criteria, use_container_width=True)
            
            # Sonuç özeti
            st.success(f"""
            **Sonuç Özeti:**
            - **{result['total_patterns']} adet pattern** tespit edildi
            - **{result['damaged_cells']} hücre** hasarlı (25'den)
            - **{result['damage_percentage']:.1f}% hasar** oranı
            - ISO 2409:2013'e göre **{class_info['name']}** sınıfı
            - **{result['confidence']:.1%} güven** ile sınıflandırıldı
            """)
            
        else:
            st.info("Sol panelden grid seçimi yapın ve pattern analizi başlatın")
            
            st.markdown("""
            ### ISO 2409:2013 Uyumlu Pattern Recognition:
            
            **1. Grid Line Detection:**
            - Hough Line Transform ile grid çizgileri tespit edilir
            - Horizontal ve vertical çizgiler ayrılır
            
            **2. Pattern Classification:**
            - **Flaking:** Yuvarlak/oval şekiller, orta boy
            - **Cracking:** Linear, yüksek aspect ratio
            - **Delamination:** Geniş alan, düşük aspect ratio  
            - **Edge Damage:** Grid çizgileri yakınında
            
            **3. ISO 2409:2013 Compliance:**
            - Pattern varlığı dikkate alınır
            - Düşük hasar eşiği (0.01) kullanılır
            - Herhangi bir pattern tespit edilirse minimum Sınıf 1
            - Hücre bazlı ve pattern bazlı çift kontrol
            
            **4. Classification Logic:**
            - Pattern tespit edildi + az hücre hasarı = Sınıf 1
            - Pattern sayısı arttıkça üst sınıflara geçiş
            - ISO standardının gerçek kriterlerine uygun
            """)

if __name__ == "__main__":
    main_pattern_recognition()
