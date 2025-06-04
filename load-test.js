import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: 1000, // número de usuarios virtuales simultáneos
  duration: '30s', // duración de la prueba
};

export default function () {
  http.get('<TU_ENDPOINT_DE_CLOUD_RUN>');
  sleep(1);
}