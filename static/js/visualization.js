import * as THREE from './three.module.js';
console.log(THREE);
// 初始化场景 
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, 600);
document.getElementById('visualization-container').appendChild(renderer.domElement);

// 添加点云
import { PLYLoader } from './PLYLoader.js';
const loader = new PLYLoader();
//const loader = new THREE.PLYLoader();
loader.load('./static/models/pcd/mars_11/point_cloud.ply', function (geometry) {
    const material = new THREE.PointsMaterial({ size: 0.05, color: 0xffffff });
    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
});

// 相机轨迹
const cameraGroup = new THREE.Group();
const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
const frustumMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 });
const cameraPoses = [];

// 加载相机位姿
fetch('./static/models/pose/directions_mars_11.txt')
    .then((response) => response.text())
    .then((data) => {
        const lines = data.split('/n');
        const positions = [];
        lines.forEach((line) => {
            if (!line.trim()) return;
            const [x, y, z, dx, dy, dz] = line.split(' ').map(parseFloat);
            positions.push(new THREE.Vector3(x, y, z));
            cameraPoses.push({ position: new THREE.Vector3(x, y, z), direction: new THREE.Vector3(dx, dy, dz) });
        });

        // 添加轨迹线
        const trajectoryGeometry = new THREE.BufferGeometry().setFromPoints(positions);
        const trajectoryLine = new THREE.Line(trajectoryGeometry, lineMaterial);
        cameraGroup.add(trajectoryLine);

        // 添加相机视锥
        cameraPoses.forEach((pose) => {
            const frustumGeometry = new THREE.BufferGeometry();
            const { position, direction } = pose;

            const fov = 45; // 相机视角
            const near = 0.1; // 近裁剪面
            const far = 1.0; // 远裁剪面
            const aspect = 1.5; // 宽高比

            const tanFOV = Math.tan((fov * Math.PI) / 360.0);

            const nearHeight = tanFOV * near;
            const nearWidth = nearHeight * aspect;

            const farHeight = tanFOV * far;
            const farWidth = farHeight * aspect;

            // 定义视锥四个角点
            const points = [
                position.clone(),
                position.clone().add(direction.clone().setLength(near)),
                position.clone().add(direction.clone().setLength(far)),
            ];

            frustumGeometry.setFromPoints(points);
            const frustum = new THREE.Line(frustumGeometry, frustumMaterial);
            cameraGroup.add(frustum);
        });
    });

scene.add(cameraGroup);

// 滑动条
const slider = document.getElementById('frame-slider');
slider.addEventListener('input', (event) => {
    const frameIndex = parseInt(event.target.value);
    // 更新相机位置
    if (cameraPoses[frameIndex]) {
        const { position, direction } = cameraPoses[frameIndex];
        camera.position.copy(position);
        camera.lookAt(position.clone().add(direction));
    }
});

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
