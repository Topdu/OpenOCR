"""
OpenOCR Test Script
Tests all OpenOCR tasks using both Python API and command-line interface.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path for imports
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)

from tools.download_example_images import download_example_images
from tools.utils.logging import get_logger

logger = get_logger(name='test_openocr')


class OpenOCRTester:
    """Test suite for OpenOCR functionality"""

    def __init__(self):
        """Initialize tester and download test images"""
        logger.info('=' * 80)
        logger.info('OpenOCR Test Suite')
        logger.info('=' * 80)

        # Download test images
        logger.info('\n📥 Downloading test images...')
        self.image_paths = download_example_images(use_modelscope=True)

        # Verify image paths
        self.ocr_images = Path(self.image_paths.get('ocr', ''))
        self.rec_images = Path(self.image_paths.get('unirec', '')) / '..' / 'rec'  # Use rec folder
        self.doc_images = Path(self.image_paths.get('doc', ''))
        self.unirec_images = Path(self.image_paths.get('unirec', ''))

        # Create output directory
        self.output_dir = Path('test_output')
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"\n📁 Test image directories:")
        logger.info(f"  OCR/Det: {self.ocr_images}")
        logger.info(f"  Rec: {self.rec_images}")
        logger.info(f"  Doc: {self.doc_images}")
        logger.info(f"  UniRec: {self.unirec_images}")
        logger.info(f"  Output: {self.output_dir}")

    def get_test_image(self, image_dir):
        """Get first valid image from directory"""
        if not image_dir.exists():
            logger.warning(f"Directory not found: {image_dir}")
            return None

        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            images = list(image_dir.glob(f'*{ext}'))
            if images:
                return str(images[0])

        logger.warning(f"No images found in: {image_dir}")
        return None

    def test_python_api(self):
        """Test OpenOCR using Python API"""
        logger.info('\n' + '=' * 80)
        logger.info('🐍 Testing Python API')
        logger.info('=' * 80)

        from openocr import OpenOCR

        # Test 1: Detection task
        logger.info('\n[Test 1/4] Testing Detection Task...')
        try:
            test_img = self.get_test_image(self.ocr_images)
            if test_img:
                openocr_det = OpenOCR(task='det', use_gpu='auto')
                results = openocr_det(image_path=test_img)
                boxes = results[0]['boxes']
                logger.info(f"✅ Detection: Found {len(boxes)} text regions")
            else:
                logger.warning('⚠️  Detection: No test image available')
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 2: Recognition task
        logger.info('\n[Test 2/4] Testing Recognition Task...')
        try:
            test_img = self.get_test_image(self.rec_images)
            if not test_img:
                # Fallback to ocr images
                test_img = self.get_test_image(self.ocr_images)

            if test_img:
                openocr_rec = OpenOCR(task='rec', mode='mobile', use_gpu='auto')
                results = openocr_rec(image_path=test_img, batch_num=1)
                text = results[0]['text']
                score = results[0]['score']
                logger.info(f"✅ Recognition: Text='{text}', Score={score:.3f}")
            else:
                logger.warning('⚠️  Recognition: No test image available')
        except Exception as e:
            logger.error(f"❌ Recognition failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 3: OCR task (detection + recognition)
        logger.info('\n[Test 3/4] Testing OCR Task (Detection + Recognition)...')
        try:
            test_img = self.get_test_image(self.ocr_images)
            if test_img:
                openocr_e2e = OpenOCR(task='ocr', mode='mobile', use_gpu='auto')
                output_path = self.output_dir / 'ocr_test'
                results, time_dicts = openocr_e2e(
                    image_path=test_img,
                    save_dir=str(output_path),
                    is_visualize=True,
                    rec_batch_num=6
                )
                logger.info(f"✅ OCR: Processed successfully, results saved to {output_path}")
            else:
                logger.warning('⚠️  OCR: No test image available')
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 4: UniRec task
        logger.info('\n[Test 4/4] Testing UniRec Task...')
        try:
            test_img = self.get_test_image(self.unirec_images)
            if test_img:
                openocr_unirec = OpenOCR(task='unirec', use_gpu='auto', auto_download=True)
                result_text, generated_ids = openocr_unirec(
                    image_path=test_img,
                    max_length=2048
                )
                logger.info(f"✅ UniRec: Generated {len(generated_ids)} tokens")
                logger.info(f"   Text preview: {result_text[:100]}..." if len(result_text) > 100 else f"   Text: {result_text}")
            else:
                logger.warning('⚠️  UniRec: No test image available')
        except Exception as e:
            logger.error(f"❌ UniRec failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 5: Doc task
        logger.info('\n[Test 5/5] Testing Doc Task...')
        try:
            test_img = self.get_test_image(self.doc_images)
            if test_img:
                openocr_doc = OpenOCR(
                    task='doc',
                    use_gpu='auto',
                    auto_download=True,
                    use_layout_detection=True
                )
                result = openocr_doc(
                    image_path=test_img,
                    layout_threshold=0.4,
                    max_length=2048
                )

                # Save results
                output_path = self.output_dir / 'doc_test'
                output_path.mkdir(exist_ok=True)
                openocr_doc.save_to_json(result, str(output_path))
                openocr_doc.save_to_markdown(result, str(output_path))

                # Only save visualization if layout_results exists
                if 'layout_results' in result:
                    openocr_doc.save_visualization(result, str(output_path))

                logger.info(f"✅ Doc: Processed successfully, results saved to {output_path}")
            else:
                logger.warning('⚠️  Doc: No test image available')
        except Exception as e:
            logger.error(f"❌ Doc failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info('\n' + '=' * 80)
        logger.info('✅ Python API tests completed')
        logger.info('=' * 80)

    def test_command_line(self):
        """Test OpenOCR using command-line interface"""
        logger.info('\n' + '=' * 80)
        logger.info('💻 Testing Command-Line Interface')
        logger.info('=' * 80)

        # Test 1: Detection task
        logger.info('\n[Test 1/4] Testing Detection Command...')
        try:
            test_img = self.get_test_image(self.ocr_images)
            if test_img:
                output_path = self.output_dir / 'cli_det'
                cmd = [
                    sys.executable, 'openocr.py',
                    '--task', 'det',
                    '--input_path', test_img,
                    '--output_path', str(output_path),
                    '--is_vis'
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=__dir__, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Detection CLI: Success")
                else:
                    logger.error(f"❌ Detection CLI failed: {result.stderr}")
            else:
                logger.warning('⚠️  Detection CLI: No test image available')
        except Exception as e:
            logger.error(f"❌ Detection CLI failed: {e}")

        # Test 2: Recognition task
        logger.info('\n[Test 2/4] Testing Recognition Command...')
        try:
            test_img = self.get_test_image(self.rec_images)
            if not test_img:
                test_img = self.get_test_image(self.ocr_images)

            if test_img:
                output_path = self.output_dir / 'cli_rec'
                cmd = [
                    sys.executable, 'openocr.py',
                    '--task', 'rec',
                    '--input_path', test_img,
                    '--output_path', str(output_path),
                    '--mode', 'mobile'
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=__dir__, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Recognition CLI: Success")
                else:
                    logger.error(f"❌ Recognition CLI failed: {result.stderr}")
            else:
                logger.warning('⚠️  Recognition CLI: No test image available')
        except Exception as e:
            logger.error(f"❌ Recognition CLI failed: {e}")

        # Test 3: OCR task
        logger.info('\n[Test 3/4] Testing OCR Command...')
        try:
            test_img = self.get_test_image(self.ocr_images)
            if test_img:
                output_path = self.output_dir / 'cli_ocr'
                cmd = [
                    sys.executable, 'openocr.py',
                    '--task', 'ocr',
                    '--input_path', test_img,
                    '--output_path', str(output_path),
                    '--is_vis',
                    '--mode', 'mobile'
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=__dir__, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ OCR CLI: Success")
                else:
                    logger.error(f"❌ OCR CLI failed: {result.stderr}")
            else:
                logger.warning('⚠️  OCR CLI: No test image available')
        except Exception as e:
            logger.error(f"❌ OCR CLI failed: {e}")

        # Test 4: UniRec task
        logger.info('\n[Test 4/4] Testing UniRec Command...')
        try:
            test_img = self.get_test_image(self.unirec_images)
            if test_img:
                output_path = self.output_dir / 'cli_unirec'
                cmd = [
                    sys.executable, 'openocr.py',
                    '--task', 'unirec',
                    '--input_path', test_img,
                    '--output_path', str(output_path),
                    '--max_length', '2048'
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=__dir__, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ UniRec CLI: Success")
                else:
                    logger.error(f"❌ UniRec CLI failed: {result.stderr}")
            else:
                logger.warning('⚠️  UniRec CLI: No test image available')
        except Exception as e:
            logger.error(f"❌ UniRec CLI failed: {e}")

        # Test 5: Doc task
        logger.info('\n[Test 5/5] Testing Doc Command...')
        try:
            test_img = self.get_test_image(self.doc_images)
            if test_img:
                output_path = self.output_dir / 'cli_doc'
                cmd = [
                    sys.executable, 'openocr.py',
                    '--task', 'doc',
                    '--input_path', test_img,
                    '--output_path', str(output_path),
                    '--save_vis',
                    '--save_json',
                    '--save_markdown',
                    '--use_layout_detection'
                ]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=__dir__, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Doc CLI: Success")
                else:
                    logger.error(f"❌ Doc CLI failed: {result.stderr}")
            else:
                logger.warning('⚠️  Doc CLI: No test image available')
        except Exception as e:
            logger.error(f"❌ Doc CLI failed: {e}")

        logger.info('\n' + '=' * 80)
        logger.info('✅ Command-line tests completed')
        logger.info('=' * 80)

    def run_all_tests(self):
        """Run all tests"""
        logger.info('\n🚀 Starting OpenOCR test suite...\n')

        # Test Python API
        self.test_python_api()

        # Test Command-line
        self.test_command_line()

        logger.info('\n' + '=' * 80)
        logger.info('🎉 All tests completed!')
        logger.info(f"📁 Test outputs saved to: {self.output_dir.absolute()}")
        logger.info('=' * 80)


def main():
    """Main test entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test OpenOCR functionality')
    parser.add_argument(
        '--test-type',
        type=str,
        default='all',
        choices=['all', 'python', 'cli'],
        help='Type of tests to run: all (default), python (API only), cli (command-line only)'
    )

    args = parser.parse_args()

    try:
        tester = OpenOCRTester()

        if args.test_type == 'all':
            tester.run_all_tests()
        elif args.test_type == 'python':
            tester.test_python_api()
        elif args.test_type == 'cli':
            tester.test_command_line()

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
