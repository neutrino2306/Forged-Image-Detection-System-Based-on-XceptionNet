// imageStore.js
import { ref, reactive } from 'vue';

export const imageSrc = ref(null);
export const detectionResult = reactive({
    isFake: null
});