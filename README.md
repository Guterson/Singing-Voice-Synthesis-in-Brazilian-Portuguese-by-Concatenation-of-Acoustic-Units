# Singing Voice Synthesis in Brazilian Portuguese by Concatenation of Acoustic Units

Graduation's final project submitted to the teaching staff from the Eletronics and Computing Engineering bachelor's course from the Polytechnic School of the Federal University of Rio de Janeiro as one of the necessary requirements for obtaining the Electronics and Computing Engineer degree.

## About

The project consisted in a deep study of the process of selecting acoustic units (syllabic-level utterances), with the proposal of a recording list, also know as *reclist*, especialy tailored to brazilian portuguese phonotactics and phonemic content. Then, using the proposed recording list, a voicebank was recorded using [OREMO](http://nwp8861.web.fc2.com/soft/oremo/index.html) in a acoustically treated recording room with high quality recording equipment. Finally, a procedural Python3 script was created for synthesizing singing voice using the recorded voicebank. All the process, from the concept and design of the recording list to the parameters used in the synthesizing stage, were heavilly based on the [UTAU](http://utau2008.web.fc2.com/index.html) software and its community culture of self-recording voicebanks.

## Contents

- A Python3 script used for synthesizing a specific audio example using the recorded voicebank. 

- A .pdf file (written in brazilian Portuguese) describing all the process and reasoning from recording a voicebank to synthesizing audio via a Python script. The original monograph was published [here](https://www.repositorio.poli.ufrj.br/monografias/projpoli10043782.pdf).

- Somes samples of the monopitch voicebank recorded by [me](https://github.com/Guterson). If you want access to the full voicebank, please contact [me](mailto:gml@poli.ufrj.br).

## Notes

- This project does not uses machine learning techniques directly, using only "classical" signal processing techniques and Python packages that doesn't require a training stage.

- For complete use of the voicebank, it's necesary to create time annotations in ".oto" files, in a process know as [otoing](http://utau.wikidot.com/tutorials:cv-otoing-guide-by-kiyoteru).

## Other Works

For similar works, please visit [my portfolio](https://github.com/Guterson/Portfolio).
