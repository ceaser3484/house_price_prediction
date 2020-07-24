## House Prices: Advanced Regression Techniques

- 주제: AI 알고리즘 활용 Ames시의 Housing dataset을 바탕으로 주택가격을 예측하기.
- 목표: 79개의 feature가 등록된 집의 정보를 바탕으로 주택 가격 예측
- 배경: Dean De Cock의 data scientist 교육을 위한 dataset입니다. 
- 주최/주관
     주최 : 제주특별자치도청, 제주테크노파크
     주관 : Kaggle
- 웹사이트 : [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

## II. 데이터 구성
### (1) 데이터 셋 구성
    - 훈련데이터:  train.csv (449.88 KB)
    - 테스트데이터: test.csv (440.83 KB)

- 2020.04 기간 내 카드 데이터 (7/28 공개)
- 2019.01 ~ 2020.03 기간 내 카드 데이터

### (2) 데이터 정의
#### 주의: 영어의 미숙으로 완벽하게 해석하지 못하였음을 알려 드립니다. 
    MSSubClass: Identifies the type of dwelling involved in the sale.	   판매 중인 주택의 타입.

        20	1-STORY 1946 & NEWER ALL STYLES                                 1층 1946년 이후 지은 새로운 스타일의 집
        30	1-STORY 1945 & OLDER                                            1층 1945년 이전 지은 오래된 집
        40	1-STORY W/FINISHED ATTIC ALL AGES                               1층 
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER                                            2층 1946년 이후에 지은 집
        70	2-STORY 1945 & OLDER                                            2층 1946년 이전에 지은 집
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

    MSZoning: Identifies the general zoning classification of the sale.     어느 지역에 집이 있는가 확인
		
       A	Agriculture                                                   농업지역
       C	Commercial                                                    상업지역
       FV	Floating Village Residential                                  공사 중인 주거지역?
       I	Industrial                                                    산업지역 공장지역
       RH	Residential High Density                                      높은 밀집 주거지역
       RL	Residential Low Density                                       낮은 밀집 주거지역
       RP	Residential Low Density Park                                  낮은 밀집의 공원이 있는 주거지역(?)
       RM	Residential Medium Density                                    중간 밀집 주거지역
	
    LotFrontage: Linear feet of street connected to property                도로와의 직선 거리

    LotArea: Lot size in square feet                                        구획의 넓이

    Street: Type of road access to property                                 주택으로 들어가는 도로의 타입

       Grvl	Gravel	                                                      자갈길
       Pave	Paved                                                         포장도로
       	
    Alley: Type of alley access to property                               자산으로 접근하는 골목길의 타입

       Grvl	Gravel                                                       자갈길
       Pave	Paved                                                        포장도로
       NA 	No alley access                                              접근로 없음
		
    LotShape: General shape of property                                  자산의 보통의 생김새(?)

       Reg	Regular	                                                    보통
       IR1	Slightly irregular                                          약간 불규칙적인
       IR2	Moderately Irregular                                        보통 불규칙적인
       IR3	Irregular                                                   불규칙적임
       
    LandContour: Flatness of the property                                     자산의 평지 여부

       Lvl	Near Flat/Level	                                                   거의 평평함
       Bnk	Banked - Quick and significant rise from street grade to building  약간의 둔턱, 꽤 주요한 고지가 거리에서 주택까지 있음
       HLS	Hillside - Significant slope from side to side                     언덕, 주요한 경사지가 있음
       Low	Depression                                                         저지대
		
    Utilities: Type of utilities available                                  편의시설의 타입
		
       AllPub	All public Utilities (E,G,W,& S)	                          모든 공공시설물이 있음
       NoSewr	Electricity, Gas, and Water (Septic Tank)                     전기, 가스, 수도 있음, 정화조
       NoSeWa	Electricity and Gas Only                                      전기, 가스만 있음
       ELO	Electricity only	                                              전기만 있음
	
    LotConfig: Lot configuration                                            지역의 구성

       Inside	Inside lot                                                   지역 안에 있음
       Corner	Corner lot                                                   지역 모퉁이에 있음
       CulDSac	Cul-de-sac                                                   막다른 길에 있음. (조용함)
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
    LandSlope: Slope of property                                           집의 경사
		
       Gtl	Gentle slope                                                     약간 기울어져 있음
       Mod	Moderate Slope	                                                 꽤 기울어져 있음
       Sev	Severe Slope                                                     심하게 기울어져 있음
	
    Neighborhood: Physical locations within Ames city limits              Ames 시의 경계에 위치해 있는 이웃들.

        이 부분은 해석할 이유가 없기에 생략합니다. 

       Blmngtn	Bloomington Heights                    
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
    Condition1: Proximity to various conditions                          근접성에 대한 다양한 조건들
	
       Artery	Adjacent to arterial street                                간선도로와 인접해 있음
       Feedr	Adjacent to feeder street	                               큰 도로의 진입로에 인접해 있음   
       Norm	Normal	                                                       보통
       RRNn	Within 200' of North-South Railroad                            North-South Railroad가 200 안에 있음
       RRAn	Adjacent to North-South Railroad                               Adjacent to North-South Railroad가 근접함
       PosN	Near positive off-site feature--park, greenbelt, etc.          긍정적인 off-site: 공원, 그린벨트 등등
       PosA	Adjacent to postive off-site feature                           긍정적인 off-site가 근접해 있음
       RRNe	Within 200' of East-West Railroad                              East-West Railroad가 200안에 있음             
       RRAe	Adjacent to East-West Railroad                                 East-West Railroad가 근접해 있음
	
    Condition2: Proximity to various conditions (if more than one is present) 근접성에 대한 다양한 조건들(1개 이상이면)
		
       Artery	Adjacent to arterial street                                간선도로와 인접해 있음                            
       Feedr	Adjacent to feeder street	                               큰 도로의 진입로에 인접해 있음   
       Norm	Normal	                                                       보통
       RRNn	Within 200' of North-South Railroad                            North-South Railroad가 200 안에 있음
       RRAn	Adjacent to North-South Railroad                               North-South Railroad가 근접해 있음
       PosN	Near positive off-site feature--park, greenbelt, etc.          긍정적인 off-site: 공원, 그린벨트 등등
       PosA	Adjacent to postive off-site feature                           긍정적인 off-site가 근접해 있음
       RRNe	Within 200' of East-West Railroad                              East-West Railroad가 200 안에 있음
       RRAe	Adjacent to East-West Railroad                                 East-West Railroad가 근접해 있음
	
    BldgType: Type of dwelling                                            주거지의 형태
		
       1Fam	Single-family Detached	                                            단독주택
       2FmCon	Two-family Conversion; originally built as one-family dwelling  원래는 한 가족만 위해 만들었으나 두 가족용으로 개조
       Duplx	Duplex                                                          한 개의 필지, 두 가구 사는 집.
       TwnhsE	Townhouse End Unit                                              ?
       TwnhsI	Townhouse Inside Unit                                           ?
	
    HouseStyle: Style of dwelling                                         주거지의 형태
	
       1Story	One story                                                     1층 집
       1.5Fin	One and one-half story: 2nd level finished                    1층과 1.5층, 
       1.5Unf	One and one-half story: 2nd level unfinished                  ?
       2Story	Two story                                                     2층 집
       2.5Fin	Two and one-half story: 2nd level finished                    ?
       2.5Unf	Two and one-half story: 2nd level unfinished                  ?
       SFoyer	Split Foyer                                                   나누어진 
       SLvl	Split Level                                                       ?
	
    OverallQual: Rates the overall material and finish of the house              전체적인 내외 마감재 진행율

        이 부분은 해석할 이유가 없기에 생략합니다. 


       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
    OverallCond: Rates the overall condition of the house                      전체적인 집의 상태

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
    YearBuilt: Original construction date                                     

    YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)   최근 리모델링한 날짜(만일 추가적인 리모델링과 추가적인 것이 없으면 건축날짜와 같다.

    RoofStyle: Type of roof                                                      지붕의 타입

       Flat	Flat                                                             납짝함
       Gable	Gable                                                        삼각지붕
       Gambrel	Gabrel (Barn)                                                각진 삼각 지붕
       Hip	Hip                                                              둥근 지붕
       Mansard	Mansard                                                      맨사드(?)
       Shed	Shed                                                             오두막
		
    RoofMatl: Roof material                                                지붕 타입

       ClyTile	Clay or Tile                                                진흙과 타일
       CompShg	Standard (Composite) Shingle                                
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
    Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
    Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
    MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone

### (3) 참고
    - 모든 데이터는 [구글 클라우드 빅쿼리]에 적재하였습니다.  그리고 아래와 같이 불러 와서 머신러닝 프로젝트를 수행

``` python
import 

sql  = ''

```

5. 참가자 대상

    - AI, 빅데이터에 관심있는 누구나 참여 가능



6. 상금/시상

    - 1위 (대상) : 300 만원 / 제주특별자치도지사상

    - 2위 (최우수상) : 200 만원 / 제주테크노파크원장상

    - 3위 (우수상) : 100 만원 / 제주테크노파크원

## III. 개발스펙
### (1) OS 환경
 - OS 환경 Linux, Jupyter, Google Colab

### (2) Python 비, 패키지 버전

