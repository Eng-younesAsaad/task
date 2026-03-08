import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================================================
# 1. إعدادات البيئة والمسارات (عدل هذا السطر فقط)
# =========================================================
# ضع مسار مجلد البيانات الرئيسي هنا
# مثال: r"C:\Users\Abdurrahman\Desktop\Project\spectrograms_data"
DATA_PATH = r"D:\My_Epilepsy_Project\spectrograms_data"  # <--- عدل المسار هنا

BATCH_SIZE_PER_FILE = 50  # كل ملف يحتوي 50 صورة
IMG_HEIGHT = 59
IMG_WIDTH = 114
IMG_CHANNELS = 22
EPOCHS = 20

# التأكد من استخدام كرت الشاشة إذا وجد
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# =========================================================
# 2. تصميم مولد البيانات (The Smart Generator - Bulletproof)
# =========================================================
class LocalDataGenerator(keras.utils.Sequence):
    def __init__(self, file_paths, labels, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        idx = self.indices[index]
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            data = np.load(file_path)
            
            # حماية 1: التأكد أن الملف ليس فارغاً
            if data.size == 0:
                return np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)), np.array([label])

            # حماية 2: توحيد الأبعاد (ضمان أن المصفوفة دائماً 4D)
            # إذا كان الملف يحتوي صورة واحدة (3D)، نضيف بعداً رابعاً ليصبح (1, 22, 59, 114)
            if data.ndim == 3:
                data = np.expand_dims(data, axis=0)

            # حماية 3: تعديل ترتيب القنوات (Channels Last)
            # يحول من (Batch, 22, 59, 114) إلى (Batch, 59, 114, 22) ليطابق Keras
            data = np.moveaxis(data, 1, -1)

            # تجهيز الليبلز بناءً على عدد الصور الفعلي في الملف (سواء كان 50 أو أقل)
            y = np.full((data.shape[0],), label, dtype='float32')

            return data, y

        except Exception as e:
            # حماية 4: في حال كان الملف تالفاً (Corrupted)، نتجاهله بهدوء ونكمل
            print(f"\n⚠️ تخطي الملف التالف {os.path.basename(file_path)}")
            # نرجع صورة سوداء (أصفار) لكي لا يتوقف التدريب، والموديل سيتجاهلها
            return np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)), np.array([label])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
# =========================================================
# 3. تجهيز قوائم الملفات
# =========================================================
print("🔍 جاري فحص الملفات...")

all_files = []
all_labels = []

# البحث في جميع مجلدات المرضى (paz01, paz02, ...)
patient_folders = glob.glob(os.path.join(DATA_PATH, "paz*"))

if not patient_folders:
    raise ValueError(f"❌ لم يتم العثور على أي مجلدات مرضى في المسار: {DATA_PATH}")

for p_folder in patient_folders:
    # 1. ملفات النوبات (Ictal) -> Label 1
    ictal_files = glob.glob(os.path.join(p_folder, "spec_I_*.npy"))
    all_files.extend(ictal_files)
    all_labels.extend([1] * len(ictal_files))

    # 2. ملفات الطبيعي (Interictal) -> Label 0
    # ملاحظة: سنأخذ عينة متوازنة إذا أردت، أو الكل
    interictal_files = glob.glob(os.path.join(p_folder, "spec_P_*.npy"))
    all_files.extend(interictal_files)
    all_labels.extend([0] * len(interictal_files))

print(f"✅ تم العثور على {len(all_files)} ملف (كل ملف يحتوي 50 صورة).")
print(f"📊 إجمالي الصور المتوقعة للتدريب: {len(all_files) * 50}")

# تقسيم البيانات (Train / Validation)
X_train_files, X_val_files, y_train, y_val = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# إنشاء المولدات
train_gen = LocalDataGenerator(X_train_files, y_train, shuffle=True)
val_gen = LocalDataGenerator(X_val_files, y_val, shuffle=False)

# =========================================================
# 4. بناء الموديل (CNN)
# =========================================================
def build_cnn_model():
    model = models.Sequential([
        # طبقة الإدخال: (59, 114, 22)
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(), # تحسين الاستقرار
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # لتقليل الـ Overfitting
        layers.Dense(1, activation='sigmoid') # تصنيف ثنائي (نوبة / لا نوبة)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_model()
model.summary()

# =========================================================
# 5. التدريب
# =========================================================
# حفظ أفضل موديل تلقائياً
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_cnn_model_fixed.keras", save_best_only=True, monitor='val_accuracy'
)

# التوقف المبكر إذا لم يتحسن الموديل
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor='val_accuracy'
)

print("\n🚀 بدء التدريب على البيانات الكاملة...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

print("\n✅ تم الانتهاء من التدريب بنجاح!")

# =========================================================
# 6. رسم النتائج
# =========================================================
# رسم الدقة والخسارة وحفظها كصورة للتقرير
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy (Corrected Data)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Corrected Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('semester1_corrected_results.png')
plt.show()

print("📈 تم حفظ رسم النتائج في ملف semester1_corrected_results.png")