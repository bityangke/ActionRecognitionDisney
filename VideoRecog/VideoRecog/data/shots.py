"""
This module implements functionalities for loading and accessing shot information
"""
import glob
import os
import fnmatch
from bs4 import BeautifulSoup as bs

class ShotsInfo:
	def __init__(self, video_name, duration_ms, fps):
		self._video_name = video_name
		self._shots_frm = []
		self._shots_ms = []
		self._fps = fps
		self._duration_ms = duration_ms
		self._duration_frm = int(float(self._duration_ms)/1000 * self._fps)

	def add_shots(self, start_f, start_t, duration_f, duration_t):
		self._shots_ms.append((start_t, duration_t))
		self._shots_frm.append((start_f, duration_f))

	def get_shots_frm(self):
		return self._shots_frm

	def get_shots_ms(self):
		return self._shots_ms

	def get_num_shots(self):
		return len(self._shots_frm)

	def get_num_frames(self):
		return self._duration_frm


class ShotsInfoMgr:
	def __init__(self):
		self.shots = {}

	def load(self, shots_folder):
		shots_xml_lst = []
		for root, dirnames, filenames in os.walk(shots_folder):
		    for filename in fnmatch.filter(filenames, '*.xml'):
		        shots_xml_lst.append(os.path.join(root, filename))
		for shots_xml in shots_xml_lst:
			soup = bs(open(shots_xml), "html.parser")
			video_path = soup.media['src']
			video_name = video_path.split('/')[-1].split('.')[0][2:]
			fps = float(soup.fps.string)
			duration_ms = int(soup.duration.string)
			shots_info = ShotsInfo(video_name, duration_ms, fps)
			shot_tags = soup.find_all('shot')
			for shot_tag in shot_tags:
				fbegin = int(shot_tag['fbegin'])
				fduration = int(shot_tag['fduration'])
				msbegin = int(shot_tag['msbegin'])
				msduration = int(shot_tag['msduration'])
				shots_info.add_shots(fbegin, msbegin, fduration, msduration)
			self.shots[video_name] = shots_info
		
	def get_num_shots(self, video_name):
		assert(video_name in self.shots)
		return self.shots[video_name].get_num_shots()

	def get_num_frames(self, video_name):
		assert(video_name in self.shots)
		return self.shots[video_name].get_num_frames()


if __name__ == '__main__':
	mgr = ShotsInfoMgr()
	print('loading shots info...')
	mgr.load('/data03/mscvproject/data/ActivityNetShot/ActivityNetShot/val')
	print('shots info loaded...')
	num_shots = mgr.get_num_shots('5fW_2c_kKfc')
	print(num_shots)


